import zmq
import time
import heapq
import signal
import logging
import threading
import multiprocessing as mp
from threading import Event, Thread
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from sglang.srt.redis_utils import RedisClient
from sglang.utils import cleanup_zmq_ipc, get_exception_traceback
from sglang.srt.utils import configure_logger, kill_parent_process
from sglang.multi_model.multi_model_server_args import MultiModelServerArgs
from sglang.multi_model.scheduling.state import ModelInstanceState, ModelState
from sglang.multi_model.utils.get_memory_pool_size import get_model_path_to_cell_size
from sglang.srt.managers.io_struct import (GenerateReqInput, ResizeChunkInput, UpdateQueueStats)

logger = logging.getLogger(__name__)

INITIAL_PREFILL_RATE = 1.0 / 2048
RATE_CALCULATE_WINDOW = 5
EXPIRE_TIME = 20
INITIAL_OUTPUT_LENGTH = 256
RUNNING_OUTPUT_LENGTH = 20
DEFAULT_PREFILL_LEN = 1024
MAX_QUEUE_LEN = 20

model_prefill_rate_dict: Dict[str, float] = {}
model_decode_rate_dict: Dict[str, float] = {}
model_output_len_dict: Dict[str, int] = defaultdict(int)
model_input_lens_dict: Dict[str, List[int]] = defaultdict(list)

last_log_time = 0
def log_info(info: str):
    global last_log_time
    """限制log速率，每秒至多一条"""
    current_time = time.time()
    if current_time - last_log_time > 1:
        logger.info(info)
        last_log_time = current_time


class RequestWrapper:
    """封装请求对象，赋值优先级"""
    def __init__(
        self,
        req: GenerateReqInput
    ):
        self.req = req
        self.model_name = req.model
        self.violate = False

    @property
    def priority(self):
        """计算优先级
        priority = arrival_time + slo - prefill_time
        到达越早，slo越紧张，优先级越低
        """
        prefill_rate = model_prefill_rate_dict.get(self.model_name, INITIAL_PREFILL_RATE)
        input_len = self.req.prompt_len if self.req.prompt_len is not None else len(self.req.text)
        profiled_prefill_time = prefill_rate * input_len
        time_rich = self.time_remain - profiled_prefill_time
        if time_rich > 0:
            return time_rich
        else:
            if not self.violate:
                logger.info(f"⏰ Request of {self.model_name}, prefill time {profiled_prefill_time} probably violate TTFT SLO.")
                self.violate = True
        return time_rich if time_rich > 0 else profiled_prefill_time
    
    @property
    def time_remain(self):
        return self.req.arrival_time - time.time() + (self.req.slo_ttft if self.req.slo_ttft is not None else 0)

    def __lt__(self, other):
        return self.priority < other.priority

    def __str__(self):
        return f"RequestWrapper(model_name={self.model_name}, priority={self.priority}, req_id={self.req.rid})"

    def __repr__(self):
        return self.__str__()


class RequestQueueManager:
    """维护多模型请求优先队列
    资源跟踪，请求准入
    """
    def __init__(self, model_cell_size_dict: Dict[str, int]):
        self.model_name_to_cell_size = model_cell_size_dict
        self.model_queues: Dict[str, List[RequestWrapper]] = defaultdict(list) # 分模型维护优先队列
        self.lock = threading.Lock() # 队列锁
        self.last_log_time = 0
        # 运行中模型显存占用占用，未体现在物理显存变化中，需要自己跟踪
        self.activating_usages: Dict[str, float] = defaultdict(float)

    def add_requests(self, reqs: List[GenerateReqInput]):
        """添加请求batch"""
        req_wrappers = [RequestWrapper(req) for req in reqs]
        with self.lock:
            for req_wrapper in req_wrappers:
                input_len = req_wrapper.req.prompt_len
                if input_len is not None: # 测试请求不计数
                    model_input_lens_dict[req_wrapper.model_name].append(input_len)
                    heapq.heappush(self.model_queues[req_wrapper.model_name], req_wrapper)
                                
    def admission_control(
        self,
        model_instance_state_dict: Dict[str, List[ModelInstanceState]],
        model_backend_queue_lens: Dict[str, int], # Engine侧队列长度
        allow_sending_when_activating: bool = False,
    ) -> Dict[str, List[GenerateReqInput]]:
        """准入控制
        Memory Track
        |—————————————————|------------------|               |
        | activated usage | activating usage | net available |
        |  tracked usage  |        available resource        |
        """
        admitted_reqs: Dict[str, List[GenerateReqInput]] = defaultdict(list)
        with self.lock:
            for model_name, queue in self.model_queues.items():
                if model_backend_queue_lens[model_name] > MAX_QUEUE_LEN:
                    log_info(f"⌚ Queuing exceeds limit, model queue: {model_backend_queue_lens[model_name]}")
                    continue
                instance = model_instance_state_dict[model_name][0]
                if instance.state in (ModelState.INACTIVE, ModelState.DEACTIVATING):
                    log_info(f"💤 {model_name} deactivated")
                    continue
                elif instance.state == ModelState.ACTIVATING and not allow_sending_when_activating:
                    log_info(f"🔕 {model_name} does not allow message while activating")
                    continue
                availiable_memory = instance.memory_pool_size - self.activating_usages[model_name]
                retracted_req_warppers = []
                # 添加正常请求
                while queue and availiable_memory > 0:
                    req_wrapper = heapq.heappop(queue)
                    profiled_cells = self._get_request_resources(req_wrapper.req)
                    if availiable_memory >= profiled_cells:
                        admitted_reqs[model_name].append(req_wrapper.req)
                        if instance.state == ModelState.ACTIVATING: self.activating_usages[model_name] += profiled_cells
                        availiable_memory -= profiled_cells
                    else:
                        retracted_req_warppers.append(req_wrapper)
                        log_info(f"😢 Resource limited, available mem: {availiable_memory}GB, needed size: {profiled_cells}GB")
                    break
                queue.extend(retracted_req_warppers)
                heapq.heapify(queue)
        return admitted_reqs

    def _get_request_resources(self, req: GenerateReqInput) -> float:
        """估计请求显存占用，单位GB"""
        cell_size = self.model_name_to_cell_size[req.model]
        input_len = req.prompt_len if req.prompt_len is not None else len(req.text)
        return cell_size * (input_len + RUNNING_OUTPUT_LENGTH) / (1 << 30)

    def pop_model_requests(self, model_name: str) -> List[GenerateReqInput]:
        """弹出指定模型所有请求"""
        if model_name not in self.model_queues: return []
        with self.lock:
            warpped_reqs = self.model_queues[model_name]
            del self.model_queues[model_name]
            return [req_wrapper.req for req_wrapper in warpped_reqs] # 返回所有原请求对象

    def clear_activating_usage(self, model_name: str):
        """清空激活中模型资源跟踪，用于模型已成功激活"""
        with self.lock:
            if model_name in self.activating_usages:
                del self.activating_usages[model_name]               

    def __len__(self) -> int:
        return sum(len(queue) for queue in self.model_queues.values())

    def __repr__(self) -> str:
        if len(self.model_queues) == 0:
            req_counts_str = ""
        else:
            req_counts_str = ", ".join([f"{model_name}: {len(queue)}" for model_name, queue in self.model_queues.items()])
        return f"RequestQueue(total_queued={self.__len__()}, {req_counts_str})"


class ModelScheduler:
    """Request生命周期
    | -------------- global controller  waiting queue -------------- | ========> |  running  queue  |
    | frontend queue |  priority  queue  | backend queue | scheduler | to Engine | prefill | decode | finish |
    """
    def __init__(self, server_args: MultiModelServerArgs, model_path_dict: Dict[str, str]):
        self.server_args = server_args
        self.model_path_dict = model_path_dict
        self.model_state_dict = {model_name: ModelState.ACTIVE for model_name in self.model_path_dict.keys()}
        model_cell_size_dict = self._init_model_cell_size_mapping(model_path_dict)
        self.queue_manager = RequestQueueManager(model_cell_size_dict) # 全部模型请求队列
        self.model_instance_state_dict = None
        # 初始化Redis服务器
        self.redis_client = RedisClient(server_args.redis_host, server_args.redis_port, server_args.redis_db)
        self._shutdown_event = Event()
        self._receiver_thread = None
        # 初始化zmq信道
        self.context = zmq.Context(io_threads=1)
        # model scheduler <=chunk resize=> scheduler
        self.recv_from_scheduler_ipc_name = "scheduler_to_model_scheduler"
        self.recv_from_scheduler = self.context.socket(zmq.PULL)
        self.recv_from_scheduler.bind(f"ipc://{self.recv_from_scheduler_ipc_name}")
        self.send_to_scheduler_ipc_name = "model_scheduler_to_scheduler"
        self.send_to_scheduler = self.context.socket(zmq.PUSH)
        self.send_to_scheduler.connect(f"ipc://{self.send_to_scheduler_ipc_name}")
        # controller =instance state=> model scheduler
        self.recv_from_controller_ipc_name = "controller_to_model_scheduler"
        self.recv_from_controller = self.context.socket(zmq.PULL)
        self.recv_from_controller.bind(f"ipc://{self.recv_from_controller_ipc_name}")
        # 请求统计窗口
        self.last_manage_time: Dict[str, float] = defaultdict(float)
        self.model_prefill_rates_window: Dict[str, List[float]] = defaultdict(list)
        self.model_decode_rates_window: Dict[str, List[float]] = defaultdict(list)
        self.model_output_lens: Dict[str, List[int]] = defaultdict(list)
        self.req_stats: Dict[str, List[float]] = defaultdict(list) # arrive_time, input_len, prefill_time, finish_time
        # 运行请求统计
        
    def _init_model_cell_size_mapping(self, model_path_dict: Dict[str, str]) -> Dict[str, int]:
        """获取model name -> cell size映射"""
        cell_sizes = get_model_path_to_cell_size(model_path_dict.values())
        return {model_name: cell_sizes[model_path] for model_name, model_path in model_path_dict.items()}
    
    def run_scheduling_loop(self):
        """调度主循环"""
        self._receiver_thread = Thread(target=self._recv_requests_loop, daemon=True)
        self._receiver_thread.start()
        try:
            while not self._shutdown_event.is_set():
                # 获取后端运行请求队列长度
                active_models = self._get_active_model_names()
                model_backend_queue_lens = {
                    model_name: self.redis_client.get_queue_length(
                        f"{self.server_args.backend_generate_request_key_prefix}:{model_name}"
                    ) for model_name in active_models
                }
                if self.model_instance_state_dict is not None:
                    # 准入控制：根据优先级
                    admitted_reqs = self.queue_manager.admission_control(
                        model_instance_state_dict=self.model_instance_state_dict,
                        model_backend_queue_lens=model_backend_queue_lens,
                        allow_sending_when_activating=True,
                    )
                    for model_name, reqs in admitted_reqs.items():
                        logger.info(f"✔ Admitted {len(reqs)} requests for {model_name}")
                    self._send_to_backend_queue(admitted_reqs)
                time.sleep(0.01)
        except Exception as e:
            logger.error(f"Error in scheduling loop: {get_exception_traceback()}")
            self.shutdown()
                    
    def _recv_requests_loop(self):
        while not self._shutdown_event.is_set():
            try:
                self._recv_from_controller() # 更新model-instance states
                self._recv_from_scheduler() # 更新队列统计信息
                reqs = self._recv_from_frontend_queue() # 接收生成请求
                self.queue_manager.add_requests(reqs) # 加入等待队列
                for req in reqs:
                    if req.prompt_len is not None: # 测试请求不计数
                        self.req_stats[req.rid] = [req.arrival_time, req.prompt_len]
            except Exception as e:
                if self._shutdown_event.is_set(): break
                logger.error(f"Receiver error: {get_exception_traceback()}")
                raise e

    def _recv_from_controller(self):
        obj = None
        while True:
            try:
                obj = self.recv_from_controller.recv_pyobj(zmq.NOBLOCK)
            except zmq.ZMQError as e:
                break
        if obj is not None:
            self.schedule(obj)
            
    def _recv_from_scheduler(self):
        messages: List[UpdateQueueStats] = []
        while True:
            try:
                obj = self.recv_from_scheduler.recv_pyobj(zmq.NOBLOCK)
                messages.append(obj)
            except zmq.ZMQError as e:
                break
        updated: Dict[str, bool] = defaultdict(bool)
        for obj in reversed(messages):
            if not updated[obj.model_name]:
                self.manage(obj)
                updated[obj.model_name] = True

    def _recv_from_frontend_queue(self) -> List[GenerateReqInput]:
        """从前端获取请求队列
        frontend -> model scheduler
        仅对活跃模型获取
        """
        recv_reqs = []
        models_can_recv = [
            model_name for model_name, model_state in self.model_state_dict.items()
            if model_state in (ModelState.ACTIVE, ModelState.ACTIVATING) # 可接收请求的状态
        ]
        for model_name in models_can_recv:
            reqs = self.redis_client.pop_all(key=f"{self.server_args.frontend_generate_request_key_prefix}:{model_name}")
            recv_reqs.extend(reqs)
        return recv_reqs
   
    def _send_to_scheduler(self, model_name: str, chunk_size: int):
        message = ResizeChunkInput(model_name, chunk_size)
        self.send_to_scheduler.send_pyobj(message, flags=zmq.NOBLOCK)
    
    def _send_to_backend_queue(self, reqs: Dict[str, List[GenerateReqInput]]):
        """向后端发送生成请求
        model scheduler -> backend
        """
        for model_name, reqs in reqs.items():
            for req in reqs:
                self.redis_client.send_pyobj(key=f"{self.server_args.backend_generate_request_key_prefix}:{model_name}", obj=req)    
    
    def _retract_reqs_to_frontend_queue(self, model_name: str, reqs: List[GenerateReqInput]):
        """关闭模型，撤回等待队列
        frontend <- model
        """
        try:
            logger.info(f"Sending {len(reqs)} queued requests of model {model_name} back to frontend queue")
            for req in reqs:
                self.redis_client.send_pyobj(key=f"{self.server_args.frontend_generate_request_key_prefix}:{model_name}", obj=req)
        except Exception as e:
            logger.error(f"Error sending preempted requests: {get_exception_traceback()}")        
        
    def _get_active_model_names(self) -> List[str]:
        results = []
        for model_name, model_state in self.model_state_dict.items():
            if model_state in (ModelState.ACTIVE, ModelState.ACTIVATING, ModelState.DEACTIVATING):
                results.append(model_name)
        return results
    
    def schedule(self, model_instance_state_dict: Dict[str, List[ModelInstanceState]]):
        # 更新模型状态，暂停并撤回关闭模型的请求
        logger.info("Updating instance states...")
        self.model_instance_state_dict = model_instance_state_dict
        for model_name, instances in model_instance_state_dict.items():
            new_state = instances[0].state
            if self.model_state_dict[model_name] != new_state:
                if new_state == ModelState.INACTIVE:
                    waiting_reqs = self.queue_manager.pop_model_requests(model_name)
                    backend_reqs = self.redis_client.pop_all(key=f"{self.server_args.backend_generate_request_key_prefix}:{model_name}")
                    retract_reqs = waiting_reqs + backend_reqs
                    if retract_reqs:
                        self._retract_reqs_to_frontend_queue(model_name, retract_reqs)
                elif self.model_state_dict[model_name] == ModelState.ACTIVATING:
                    self.queue_manager.clear_activating_usage(model_name)
                self.model_state_dict[model_name] = new_state
    
    def manage(self, scheduler_stats: UpdateQueueStats):
        model_name = scheduler_stats.model_name
        if time.time() - self.last_manage_time[model_name] > EXPIRE_TIME:
            self.model_prefill_rates_window[model_name].clear()
            self.model_decode_rates_window[model_name].clear()
        # resize_info = self.resize_chunk(model_name, scheduler_stats.inflight_reqs, scheduler_stats.running_reqs, scheduler_stats.chunk_size)
        for rid, prefill_timestamp in scheduler_stats.prefill_timestamps.items():
            if len(self.req_stats[rid]) == 2:
                self.req_stats[rid].append(prefill_timestamp)
                self.model_prefill_rates_window[model_name].append((prefill_timestamp - self.req_stats[rid][0]) / self.req_stats[rid][1])
        for rid, (prefill_timestamp, output_len, finish_timestamp) in scheduler_stats.finish_timestamps.items():
            if self.req_stats[rid] and len(self.req_stats[rid]) < 4:
                if len(self.req_stats[rid]) == 2:
                    self.req_stats[rid].append(prefill_timestamp)
                    self.model_prefill_rates_window[model_name].append((prefill_timestamp - self.req_stats[rid][0]) / self.req_stats[rid][1])
                elif len(self.req_stats[rid]) == 3:
                    assert self.req_stats[rid][2] == prefill_timestamp
                self.req_stats[rid].append(finish_timestamp)
                self.model_decode_rates_window[model_name].append((finish_timestamp - prefill_timestamp) / output_len)
                self.model_output_lens[model_name].append(output_len)
        self.model_prefill_rates_window[model_name] = self.model_prefill_rates_window[model_name][-RATE_CALCULATE_WINDOW:]
        self.model_decode_rates_window[model_name] = self.model_decode_rates_window[model_name][-RATE_CALCULATE_WINDOW:]
        self.last_manage_time[model_name] = time.time()
        avg_prefill_rate = (
            sum(self.model_prefill_rates_window[model_name]) / len(self.model_prefill_rates_window[model_name])
            if self.model_prefill_rates_window[model_name] else -1
        )
        avg_decode_rate = (
            sum(self.model_decode_rates_window[model_name]) / len(self.model_decode_rates_window[model_name])
            if self.model_decode_rates_window[model_name] else -1
        )
        avg_output_len = (
            sum(self.model_output_lens[model_name]) / len(self.model_output_lens[model_name])
            if self.model_output_lens[model_name] else -1
        )
        if avg_prefill_rate > 0: model_prefill_rate_dict[model_name] = avg_prefill_rate
        if avg_output_len > 0: model_output_len_dict[model_name] = int(avg_output_len)
        log_info(
            f"Updated {model_name} stats: "
            f"avg prefill rate: {avg_prefill_rate} "
            f"avg decode rate: {avg_decode_rate} "
            f"avg output length: {avg_output_len} "
            f"chunked prefill size: {scheduler_stats.chunk_size} "
            # f"{resize_info}"
        )
    
    def resize_chunk(self, model_name: str, inflight_reqs: int, running_reqs: int, chunk_size: int):
        """调整Prefill chunk大小"""
        if running_reqs == 0 or model_output_len_dict[model_name] == 0: # 冷启动：全部为prefill
            if chunk_size < 512:
                self._send_to_scheduler(model_name, 512)
                return f"prefill reqs: {inflight_reqs} decode reqs: {running_reqs} resize prefill chunk to 512"
            return ""
        prefill_ratio = inflight_reqs / (inflight_reqs + running_reqs)
        token_ratio = 1 if prefill_ratio == 1 else (prefill_ratio * chunk_size / (1 - prefill_ratio))
        avg_ratio = sum(model_input_lens_dict[model_name]) / len(model_input_lens_dict[model_name]) / model_output_len_dict[model_name]
        if token_ratio < avg_ratio / 2 and chunk_size < 512: # Prefill占比太小
            self._send_to_scheduler(model_name, 512)
            return f"prefill reqs: {inflight_reqs} decode reqs: {running_reqs} resize prefill chunk to 512"
        elif token_ratio > avg_ratio * 2 and chunk_size > 64: # Decode占比太小
            self._send_to_scheduler(model_name, 64)
            return f"prefill reqs: {inflight_reqs} decode reqs: {running_reqs} resize prefill chunk to 64"
        elif token_ratio > avg_ratio * 2 / 3 and token_ratio < avg_ratio * 3 / 2 and chunk_size != 256: # 占比接近
            self._send_to_scheduler(model_name, 256)
            return f"prefill reqs: {inflight_reqs} decode reqs: {running_reqs} resize prefill chunk to 256"
        return ""
            
    def shutdown(self):
        """关停model scheduler"""
        self._shutdown_event.set()
        try:
            if hasattr(self, "redis_client") and self.redis_client is not None:
                try:
                    self.redis_client.close()
                except Exception as e:
                    logger.warning(f"Error closing Redis client during shutdown: {e}")
        except Exception as e:
            logger.warning(f"Error during redis_client shutdown: {e}")
        if self._receiver_thread and self._receiver_thread.is_alive():
            self._receiver_thread.join(timeout=2)
        self.cleanup()
            
    def cleanup(self):
        """清理sockets"""
        try:
            zmq_sockets = {}
            if hasattr(self, "send_to_scheduler"):
                zmq_sockets["send_to_scheduler"] = self.send_to_scheduler
            if hasattr(self, "recv_from_controller"):
                zmq_sockets["recv_from_controller"] = self.recv_from_controller
            ipc_files = set()
            if hasattr(self, "send_to_scheduler_ipc_name"):
                ipc_files.add(f"ipc://{self.send_to_scheduler_ipc_name}")
            if hasattr(self, "recv_from_controller_ipc_name"):
                ipc_files.add(f"ipc://{self.recv_from_controller_ipc_name}")
            cleanup_zmq_ipc(zmq_sockets=zmq_sockets, ipc_files=ipc_files, component_name="ModelScheduler")
            if hasattr(self, "context"):
                try: self.context.term()
                except Exception as e:
                    logger.warning(f"Error terminating ZMQ context: {e}")
        except Exception as e:
            logger.error(f"Error during cleanup for model scheduler: {e}")
            
def run_model_scheduler_process(
    server_args: MultiModelServerArgs,
    model_path_dict: Dict[str, str],
    pipe_finish_writer: Optional[mp.connection.Connection] = None
):
    model_scheduler = None
    configure_logger(
        server_args,
        prefix=f" Model_Scheduler",
        log_file_suffix="model_scheduler",
    )
    logger.info(f"starting GPU scheduler")
    try:
        model_scheduler = ModelScheduler(server_args, model_path_dict)
        pipe_finish_writer.send("ready")
        def signal_handler(signum, frame):
            logger.info("Received signal to shutdown")
            if model_scheduler:
                model_scheduler.shutdown()
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        model_scheduler.run_scheduling_loop()
    except Exception:
        logger.error(get_exception_traceback())
        if model_scheduler: model_scheduler.shutdown()
        kill_parent_process()