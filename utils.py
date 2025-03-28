import logging
import os
import sys
from datetime import datetime

from tqdm import tqdm


def init_logger(log_dir: str):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'run_{timestamp}.log')

    # 设置 logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清除已有 handler，避免重复打印
    if logger.hasHandlers():
        logger.handlers.clear()

    # 输出到文件
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # 输出到控制台
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info(f"[Logger] Initialization completed：{log_path}")


def log_tqdm(iterable, desc=None):
    return tqdm(iterable, desc=desc, file=sys.stdout, disable=not sys.stdout.isatty(), leave=False)


def progress(iterable, desc="", log_interval=10):
    """
    智能进度显示器：
    - 本地交互环境下使用 tqdm
    - 非交互环境下用 logging 输出每隔 log_interval 次进度
    """
    total = len(iterable) if hasattr(iterable, '__len__') else None
    if sys.stdout.isatty() or 'PYCHARM_HOSTED' in os.environ:
        yield from tqdm(iterable, desc=desc, file=sys.stdout, leave=False)
    else:
        logging.info(f"[progress] {desc}")
        for i, item in enumerate(iterable):
            if i % log_interval == 0 or i == total - 1:
                logging.info(f"[progress] {desc}")
            yield item
