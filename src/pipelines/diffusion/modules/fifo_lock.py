import threading
import collections


# reference: https://gist.github.com/vitaliyp/6d54dd76ca2c3cdfc1149d33007dc34a
class FIFOLock(object):
    """
    一个基于队列的先进先出 (FIFO) 锁
    """
    def __init__(self):
        """
        初始化方法，创建了两个线程锁 (_lock 和 _inner_lock) 和一个双向队列 (_pending_threads)。
        """
        self._lock = threading.Lock()
        self._inner_lock = threading.Lock()
        self._pending_threads = collections.deque()

    def acquire(self, blocking=True):
        """
        获取锁的方法。首先尝试获取主锁 (_lock)，如果获取成功直接返回 True。如果未能获取锁且 blocking 参数为 False，则直接返回 False。
        如果未能获取锁且 blocking 参数为 True，则创建一个 threading.Event 对象 (release_event)，
        将其加入待处理线程队列 (_pending_threads) 中，并等待该事件的触发。等待完成后再次尝试获取主锁并返回结果。
        :param blocking:
        :return:
        """
        with self._inner_lock:
            lock_acquired = self._lock.acquire(False)
            if lock_acquired:
                return True
            elif not blocking:
                return False

            release_event = threading.Event()
            self._pending_threads.append(release_event)

        release_event.wait()
        return self._lock.acquire()

    def release(self):
        """
        释放锁的方法。首先检查待处理线程队列是否有等待的线程，如果有则从队列中取出一个线程的事件 (release_event) 并触发该事件。然后释放主锁。
        :return:
        """
        with self._inner_lock:
            if self._pending_threads:
                release_event = self._pending_threads.popleft()
                release_event.set()

            self._lock.release()

    __enter__ = acquire   # 将 acquire 方法赋值给 __enter__ 方法，这样可以通过 with 语句来使用该锁。

    def __exit__(self, t, v, tb):
        """
        退出方法，在 with 语句结束时自动调用，调用 release 方法释放锁。
        :param t:
        :param v:
        :param tb:
        :return:
        """
        self.release()
