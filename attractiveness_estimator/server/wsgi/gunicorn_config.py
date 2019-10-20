from os import kill, getenv
from signal import SIGINT
from psutil import Process
from threading import Timer


MAX_RSS = 1000  # [MB]
MEM_CHECK_INTERVAL = 5 * 60  # [s]

workers = 1
timeout = 180
graceful_timeout = 10 if getenv("FLASK_ENV") == "production" else 0
log_level = "INFO"
worker_class = "gevent"


def post_worker_init(worker):
    process = Process(worker.pid)

    def mem_monitor():
        rss = process.memory_info().rss / 1000 / 1000  # [MB]
        if rss > MAX_RSS:
            sig = SIGINT
            worker.log.info(
                "Workers RSS > Max RSS ({:.2f} MB > {:.2f} MB)".format(rss, MAX_RSS))
            worker.log.info(
                "Suicide with signal {}".format(sig))
            kill(worker.pid, sig)

    t = Timer(MEM_CHECK_INTERVAL, mem_monitor)
    t.name = "WorkerMemoryMonitorTimer"
    t.daemon = True
    t.start()
