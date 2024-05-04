import time
import argparse


class TimerSubcategory:
    """
    用于记录 Timer 类中的子类别的时间。它实现了上下文管理器协议，允许使用 with 语句来管理计时的开始和结束。
    """
    def __init__(self, timer, category):
        """
        初始化一个子类别计时器，需要传入一个 Timer 实例和子类别的类别名称。
        :param timer:
        :param category:
        """
        self.timer = timer
        self.category = category
        self.start = None
        self.original_base_category = timer.base_category

    def __enter__(self):
        """
        进入上下文时调用的方法，记录开始时间，并更新 Timer 实例的 base_category 和 subcategory_level，如果需要打印日志，则打印子类别名称。
        :return:
        """
        self.start = time.time()
        self.timer.base_category = self.original_base_category + self.category + "/"
        self.timer.subcategory_level += 1

        if self.timer.print_log:
            print(f"{'  ' * self.timer.subcategory_level}{self.category}:")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        退出上下文时调用的方法，计算子类别的时间并更新记录。然后恢复 Timer 实例的 base_category 和 subcategory_level，
        最后调用 Timer 实例的 record 方法记录子类别的时间。
        :param exc_type:
        :param exc_val:
        :param exc_tb:
        :return:
        """
        elapsed_for_subcategroy = time.time() - self.start
        self.timer.base_category = self.original_base_category
        self.timer.add_time_to_record(self.original_base_category + self.category, elapsed_for_subcategroy)
        self.timer.subcategory_level -= 1
        self.timer.record(self.category, disable_log=True)


class Timer:
    def __init__(self, print_log=False):
        """
        初始化计时器，可以选择是否打印日志
        :param print_log: whether to print log
        """
        self.start = time.time()
        self.records = {}    # 记录各个类别的时间
        self.total = 0    # 总时间
        self.base_category = ''
        self.print_log = print_log
        self.subcategory_level = 0

    def elapsed(self):
        """
        :return: start距离当前的时间
        """
        end = time.time()
        res = end - self.start
        self.start = end
        return res

    def add_time_to_record(self, category, amount):
        """
        将指定类别的时间增加 amount
        :param category:
        :param amount:
        :return:
        """
        if category not in self.records:
            self.records[category] = 0

        self.records[category] += amount

    def record(self, category, extra_time=0, disable_log=False):
        """
        记录一个类别的时间，并可选地增加额外时间
        :param category:
        :param extra_time:
        :param disable_log:
        :return:
        """
        e = self.elapsed()

        self.add_time_to_record(self.base_category + category, e + extra_time)

        self.total += e + extra_time

        if self.print_log and not disable_log:
            print(f"{'  ' * self.subcategory_level}{category}: done in {e + extra_time:.3f}s")

    def subcategory(self, name):
        """
        返回一个 TimerSubcategory 实例，用于记录子类别的时间。
        :param name:
        :return:
        """
        self.elapsed()

        subcat = TimerSubcategory(self, name)
        return subcat

    def summary(self):
        """
        :return:一个字符串，包含总时间和超过0.1秒的类别及其时间
        """
        res = f"{self.total:.1f}s"

        additions = [(category, time_taken) for category, time_taken in self.records.items() if time_taken >= 0.1 and '/' not in category]
        if not additions:
            return res

        res += " ("
        res += ", ".join([f"{category}: {time_taken:.1f}s" for category, time_taken in additions])
        res += ")"

        return res

    def dump(self):
        """
        :return:返回一个字典，包含总时间和各个类别的时间。
        """
        return {'total': self.total, 'records': self.records}

    def reset(self):
        """
        重置计时器，重新初始化各个属性。
        :return:
        """
        self.__init__()


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--log-startup", action='store_true', help="print a detailed log of what's happening at startup")
args = parser.parse_known_args()[0]

startup_timer = Timer(print_log=args.log_startup)

startup_record = None
