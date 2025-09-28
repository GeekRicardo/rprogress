import random
import sys
import time
import typing
import re
from typing import (
    BinaryIO,
    Callable,
    Iterable,
    Iterator,
    List,
    NewType,
    Optional,
    Sequence,
    TextIO,
    TypeVar,
    Union,
    Any,
)

from rich.console import Console
from rich.style import StyleType
from rich.text import Text
from rich import filesize

from rich.progress import (
    Progress as RichProgress,
    BarColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TextColumn,
    ProgressColumn,
    Task,
)

ProgressType = TypeVar("ProgressType")


class SpeedColumn(ProgressColumn):
    @classmethod
    def render_speed(cls, speed: Optional[float]) -> Text:
        if speed is None:
            return Text("-- it/s", style="progress.percentage")
        unit, suffix = filesize.pick_unit_and_suffix(
            int(speed),
            ["", "×10³", "×10⁶", "×10⁹", "×10¹²"],
            1000,
        )
        data_speed = speed / unit
        return Text(f"{data_speed:.1f}{suffix} it/s", style="progress.percentage")

    def render(self, task: "Task") -> Text:
        return self.render_speed(task.finished_speed or task.speed)


class ProgressManager:
    """管理共享的进度条实例"""
    _instance: Optional['ProgressManager'] = None
    _progress: Optional[RichProgress] = None
    _active_count: int = 0
    _console: Optional[Console] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_progress(
        self,
        auto_refresh: bool = True,
        console: Optional[Console] = None,
        transient: bool = False,
        get_time: Optional[Callable[[], float]] = None,
        refresh_per_second: float = 10,
        style: StyleType = "bar.back",
        complete_style: StyleType = "bar.complete",
        finished_style: StyleType = "bar.finished",
        pulse_style: StyleType = "bar.pulse",
        disable: bool = False,
    ) -> RichProgress:
        """获取或创建共享的 RichProgress 实例"""
        if self._progress is None:
            columns: List["ProgressColumn"] = [
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(
                    style=style,
                    complete_style=complete_style,
                    finished_style=finished_style,
                    pulse_style=pulse_style,
                ),
                TaskProgressColumn(show_speed=True),
                SpeedColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(elapsed_when_finished=True),
            ]
            
            self._console = console or Console()
            self._progress = RichProgress(
                *columns,
                auto_refresh=auto_refresh,
                console=self._console,
                transient=transient,
                get_time=get_time,
                refresh_per_second=refresh_per_second,
                disable=disable,
            )
        return self._progress
    
    def start(self):
        """启动进度条（如果还没启动）"""
        if self._active_count == 0 and self._progress is not None:
            self._progress.__enter__()
        self._active_count += 1
    
    def stop(self):
        """停止进度条（如果没有其他活动任务）"""
        self._active_count -= 1
        if self._active_count == 0 and self._progress is not None:
            self._progress.__exit__(None, None, None)
            self._progress = None
            self._console = None


# 全局进度管理器
_progress_manager = ProgressManager()


def safe_get_attr(obj: Any, path: str, default: Any = None) -> Any:
    """
    安全地获取对象的嵌套属性
    支持：
    - 点号访问: obj.attr.subattr
    - 方括号访问: obj[key][0]
    - 混合访问: obj.attr[key].subattr
    - 方法调用: obj.method()
    
    Args:
        obj: 要访问的对象
        path: 访问路径
        default: 默认值
    
    Returns:
        属性值或默认值
    """
    if obj is None:
        return default
    
    # 如果path为空，返回obj本身
    if not path:
        return obj
    
    try:
        current = obj
        i = 0
        path_len = len(path)
        
        while i < path_len:
            # 跳过开头的点号
            if path[i] == '.':
                i += 1
                continue
            
            # 处理方括号访问
            if path[i] == '[':
                # 找到对应的右括号
                j = i + 1
                bracket_count = 1
                while j < path_len and bracket_count > 0:
                    if path[j] == '[':
                        bracket_count += 1
                    elif path[j] == ']':
                        bracket_count -= 1
                    j += 1
                
                if bracket_count != 0:
                    return default
                    
                key = path[i+1:j-1].strip()
                
                # 去除引号
                if (key.startswith('"') and key.endswith('"')) or \
                   (key.startswith("'") and key.endswith("'")):
                    key = key[1:-1]
                
                # 尝试转换为整数
                try:
                    key = int(key)
                except ValueError:
                    pass
                
                # 访问元素
                if isinstance(current, dict):
                    current = current.get(key, default)
                    if current is default:
                        return default
                elif isinstance(current, (list, tuple)):
                    try:
                        current = current[key]
                    except (IndexError, TypeError, KeyError):
                        return default
                else:
                    try:
                        current = current[key]
                    except:
                        return default
                
                i = j
                
            # 处理属性访问或方法调用
            else:
                # 找到下一个分隔符
                j = i
                while j < path_len and path[j] not in '.[](':
                    j += 1
                
                # 检查是否是方法调用
                if j < path_len and path[j] == '(':
                    # 找到对应的右括号
                    paren_count = 1
                    k = j + 1
                    while k < path_len and paren_count > 0:
                        if path[k] == '(':
                            paren_count += 1
                        elif path[k] == ')':
                            paren_count -= 1
                        k += 1
                    
                    if paren_count == 0:
                        method_name = path[i:j]
                        # 获取方法
                        if hasattr(current, method_name):
                            method = getattr(current, method_name)
                            if callable(method):
                                try:
                                    # 简单起见，只支持无参数调用
                                    current = method()
                                except:
                                    return default
                            else:
                                current = method
                        else:
                            return default
                        i = k
                    else:
                        return default
                else:
                    token = path[i:j]
                    
                    # 普通属性访问
                    if isinstance(current, dict):
                        current = current.get(token, default)
                        if current is default:
                            return default
                    elif hasattr(current, token):
                        current = getattr(current, token)
                    else:
                        return default
                    
                    i = j
        
        return current if current is not None else default
        
    except Exception as e:
        return default


def format_description(template: str, item: Any, index: int = 0, total: Optional[int] = None) -> str:
    """
    格式化描述字符串，支持复杂的嵌套访问
    
    Args:
        template: 描述模板，支持复杂的占位符如 {item.user.name}, {item[data][0].value}
        item: 当前迭代的元素
        index: 当前索引
        total: 总数
    
    Returns:
        格式化后的描述字符串
    """
    if not template:
        return ""
    
    # 创建基础上下文
    context = {
        'item': item,
        'index': index,
        'total': total,
        'i': index,  # 简写
        't': total,  # 简写
    }
    
    # 如果item为None，返回模板
    if item is None:
        try:
            return template.format(**context)
        except:
            return template
    
    # 匹配所有占位符内容
    placeholder_pattern = r'\{([^{}]+)\}'
    
    def replace_placeholder(match):
        placeholder = match.group(1).strip()
        
        # 如果是基础context中的键，直接返回
        if placeholder in context:
            value = context[placeholder]
            # 特殊处理：如果是item且后面没有属性访问，直接返回
            if placeholder == 'item':
                return str(value)
            return str(value)
        
        # 处理item的方法调用（如 item.upper()）
        if placeholder.startswith('item.') or placeholder.startswith('item[') or placeholder.startswith('item('):
            if placeholder.startswith('item.'):
                path = placeholder[5:]  # 去掉 'item.'
            elif placeholder.startswith('item('):
                # item本身的方法调用
                path = placeholder[4:]  # 去掉 'item'
            else:
                path = placeholder[4:]  # 去掉 'item'
            
            value = safe_get_attr(item, path)
            if value is None and value != 0 and value != False and value != '':
                # 如果是item本身的方法调用，尝试直接调用
                if placeholder.startswith('item.') and '(' in placeholder:
                    method_name = placeholder[5:].split('(')[0]
                    if hasattr(item, method_name):
                        try:
                            method = getattr(item, method_name)
                            if callable(method):
                                value = method()
                        except:
                            pass
                
                if value is None:
                    return f"{{{placeholder}}}"
            return str(value)
        
        # 尝试直接从item获取（支持 meta.badges[0] 这种格式）
        value = safe_get_attr(item, placeholder)
        if value is not None or value == 0 or value == False or value == '':
            return str(value)
        
        # 如果item是字典，再尝试一种解析方式
        if isinstance(item, dict):
            # 找到第一个键
            first_dot = placeholder.find('.')
            first_bracket = placeholder.find('[')
            
            if first_dot == -1 and first_bracket == -1:
                # 简单键
                if placeholder in item:
                    return str(item[placeholder])
            else:
                # 复杂路径
                if first_bracket != -1 and (first_dot == -1 or first_bracket < first_dot):
                    first_key = placeholder[:first_bracket]
                elif first_dot != -1:
                    first_key = placeholder[:first_dot]
                else:
                    first_key = placeholder
                
                if first_key in item:
                    remaining_path = placeholder[len(first_key):]
                    if remaining_path:
                        value = safe_get_attr(item[first_key], remaining_path)
                        if value is not None or value == 0 or value == False or value == '':
                            return str(value)
                    else:
                        return str(item[first_key])
        
        # 如果item是基本类型且有方法调用
        if '(' in placeholder and ')' in placeholder:
            method_name = placeholder.split('(')[0]
            if hasattr(item, method_name):
                try:
                    method = getattr(item, method_name)
                    if callable(method):
                        value = method()
                        return str(value)
                except:
                    pass
        
        # 都失败了，返回原占位符
        return f"{{{placeholder}}}"
    
    # 使用正则表达式替换所有占位符
    try:
        result = re.sub(placeholder_pattern, replace_placeholder, template)
        return result
    except Exception:
        return template




def track(
    sequence: Union[Sequence[ProgressType], Iterable[ProgressType]],
    description: str = "Working...",
    total: Optional[float] = None,
    auto_refresh: bool = True,
    console: Optional[Console] = None,
    transient: bool = False,
    get_time: Optional[Callable[[], float]] = None,
    refresh_per_second: float = 10,
    style: StyleType = "bar.back",
    complete_style: StyleType = "bar.complete",
    finished_style: StyleType = "bar.finished",
    pulse_style: StyleType = "bar.pulse",
    update_period: float = 0.1,
    disable: bool = False,
    show_speed: bool = True,
    update_description: bool = True,
) -> Iterator[ProgressType]:
    """
    追踪可迭代对象的进度，支持复杂数据类型和嵌套属性访问
    
    Args:
        sequence: 要迭代的序列
        description: 进度条描述，支持复杂占位符：
            - {item}: 当前项
            - {index} 或 {i}: 当前索引
            - {total} 或 {t}: 总数
            - {item.attr}: 访问属性
            - {item[key]}: 访问字典键或列表索引
            - {item.attr[0].name}: 混合访问
            - {item.method()}: 调用方法
        total: 总步数
        update_description: 是否动态更新描述
        其他参数同 rich.progress.Progress
    
    Example:
        >>> # 处理嵌套字典
        >>> data = [{"user": {"name": "Alice", "age": 25}}]
        >>> for item in track(data, "Processing {item.user.name}"):
        ...     process(item)
        
        >>> # 处理复杂对象
        >>> for item in track(objects, "Item {item.data[0].value}"):
        ...     process(item)
    """
    # 获取共享的进度条实例
    progress = _progress_manager.get_progress(
        auto_refresh=auto_refresh,
        console=console,
        transient=transient,
        get_time=get_time,
        refresh_per_second=refresh_per_second,
        style=style,
        complete_style=complete_style,
        finished_style=finished_style,
        pulse_style=pulse_style,
        disable=disable,
    )
    
    # 计算总数
    if total is None:
        try:
            total = len(sequence)
        except (TypeError, AttributeError):
            total = None
    
    # 检查描述是否包含占位符
    has_placeholders = '{' in description and '}' in description
    should_update = update_description and has_placeholders
    
    # 启动进度条
    _progress_manager.start()
    
    try:
        # 添加任务
        initial_desc = description if not should_update else format_description(description, None, 0, total)
        task_id = progress.add_task(initial_desc, total=total)
        
        # 迭代序列
        for index, item in enumerate(sequence):
            # 更新描述
            if should_update:
                formatted_desc = format_description(description, item, index, total)
                progress.update(task_id, description=formatted_desc)
            
            yield item
            progress.update(task_id, advance=1)
            
    finally:
        # 停止进度条
        _progress_manager.stop()


class Progress:
    """增强的Progress类，支持复杂数据类型和嵌套属性访问"""
    def __init__(
        self,
        sequence: Union[Sequence[ProgressType], Iterable[ProgressType]] = [],
        description: str = "Working...",
        total: Optional[float] = None,
        auto_refresh: bool = True,
        console: Optional[Console] = None,
        transient: bool = False,
        get_time: Optional[Callable[[], float]] = None,
        refresh_per_second: float = 10,
        style: StyleType = "bar.back",
        complete_style: StyleType = "bar.complete",
        finished_style: StyleType = "bar.finished",
        pulse_style: StyleType = "bar.pulse",
        update_period: float = 0.1,
        disable: bool = False,
        show_speed: bool = True,
        other_columns: List["ProgressColumn"] = [],
        shared: bool = True,
        update_description: bool = True,
    ):
        self.description = description
        self.description_template = description
        self.sequence = sequence
        self.total = (
            total or len(self.sequence)
            if not isinstance(self.sequence, Iterator)
            else None
        )
        self.update_period = update_period
        self.shared = shared
        self.task = None
        self._own_progress = False
        self.update_description = update_description
        self._current_item = None
        self._current_index = 0

        if shared:
            self.progress = _progress_manager.get_progress(
                auto_refresh=auto_refresh,
                console=console,
                transient=transient,
                get_time=get_time,
                refresh_per_second=refresh_per_second,
                style=style,
                complete_style=complete_style,
                finished_style=finished_style,
                pulse_style=pulse_style,
                disable=disable,
            )
        else:
            columns.extend(
                (
                    BarColumn(
                        style=style,
                        complete_style=complete_style,
                        finished_style=finished_style,
                        pulse_style=pulse_style,
                    ),
                    TaskProgressColumn(show_speed=True),
                    SpeedColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(elapsed_when_finished=True),
                )
            )
            columns.extend(other_columns)
            
            self.progress = RichProgress(
                *columns,
                auto_refresh=auto_refresh,
                console=console,
                transient=transient,
                get_time=get_time,
                refresh_per_second=refresh_per_second or 10,
                disable=disable,
            )
            self._own_progress = True
            
        self.tasks = self.progress._tasks

    @property
    def desc(self):
        return self.description

    def update(self, task_id, advance=1, **kwargs):
        self.progress.update(task_id, advance=advance, **kwargs)

    @desc.setter
    def desc(self, desc):
        """设置描述，支持复杂格式化模板"""
        self.description_template = desc
        if self.task is not None:
            if self._current_item is not None and self.update_description:
                formatted_desc = format_description(desc, self._current_item, self._current_index, self.total)
                self.progress.update(self.task, description=formatted_desc)
            else:
                self.progress.update(self.task, description=desc)
        self.description = desc

    def set_description(self, desc: str, refresh: bool = True):
        """设置描述的便捷方法"""
        self.desc = desc
        if refresh and self.task is not None:
            self.progress.refresh()

    def __iter__(self):
        if self.shared:
            _progress_manager.start()
        elif self._own_progress:
            self.progress.__enter__()
            
        try:
            # 检查描述是否包含占位符
            has_placeholders = '{' in self.description_template and '}' in self.description_template
            should_update = self.update_description and has_placeholders
            
            # 初始描述
            initial_desc = self.description_template
            if should_update:
                initial_desc = format_description(self.description_template, None, 0, self.total)
            
            self.task = self.progress.add_task(initial_desc, total=self.total)
            
            for index, item in enumerate(self.sequence):
                self._current_item = item
                self._current_index = index
                
                # 更新描述
                if should_update:
                    formatted_desc = format_description(self.description_template, item, index, self.total)
                    self.progress.update(self.task, description=formatted_desc)
                
                self.progress.update(self.task, advance=1)
                yield item
                
        finally:
            self._current_item = None
            self._current_index = 0
            if self.shared:
                _progress_manager.stop()
            elif self._own_progress:
                self.progress.__exit__(None, None, None)

    def __enter__(self):
        if self.shared:
            _progress_manager.start()
        elif self._own_progress:
            self.progress.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.shared:
            _progress_manager.stop()
        elif self._own_progress:
            self.progress.__exit__(exc_type, exc_val, exc_tb)

    def add_task(self, description: str, total: Optional[float] = None, **kwargs):
        return self.progress.add_task(description, total=total, **kwargs)


if __name__ == "__main__":
    import threading
    from dataclasses import dataclass
    from typing import List, Dict
    
    def process(it):
        # time.sleep(random.randint(0, 10) / 100)
        time.sleep(1)
        return it

    # 示例1：简单嵌套字典
    print("示例1：处理嵌套字典")
    users = [
        {"id": 1, "profile": {"name": "Alice", "age": 25, "city": "北京"}},
        {"id": 2, "profile": {"name": "Bob", "age": 30, "city": "上海"}},
        {"id": 3, "profile": {"name": "Charlie", "age": 35, "city": "广州"}},
    ]
    for user in track(users, "处理用户 {profile.name} ({profile.age}岁) - 城市: {profile.city}"):
        process(user)
    
    # 示例2：多层嵌套
    print("\n示例2：多层嵌套数据")
    complex_data = [
        {
            "order": {
                "id": "ORD001",
                "customer": {
                    "name": "张三",
                    "address": {
                        "city": "北京",
                        "street": "长安街1号"
                    }
                },
                "items": [
                    {"name": "商品A", "price": 100},
                    {"name": "商品B", "price": 200}
                ]
            }
        },
        {
            "order": {
                "id": "ORD002",
                "customer": {
                    "name": "李四",
                    "address": {
                        "city": "上海",
                        "street": "南京路2号"
                    }
                },
                "items": [
                    {"name": "商品C", "price": 300}
                ]
            }
        }
    ]
    for data in track(complex_data, "订单 {order.id} - 客户: {order.customer.name} ({order.customer.address.city})"):
        process(data)
    
    # 示例3：混合访问（点号和方括号）
    print("\n示例3：混合访问模式")
    mixed_data = [
        {
            "user": "Alice",
            "scores": [95, 87, 92],
            "meta": {"level": "高级", "badges": ["金牌", "银牌"]}
        },
        {
            "user": "Bob",
            "scores": [78, 85, 88],
            "meta": {"level": "中级", "badges": ["铜牌"]}
        }
    ]
    for item in track(mixed_data, "用户 {user} - 首个分数: {scores[0]} - 等级: {meta.level} - 首个徽章: {meta.badges[0]}"):
        process(item)
    
    # 示例4：使用数据类（dataclass）
    print("\n示例4：处理数据类对象")
    
    @dataclass
    class Address:
        city: str
        street: str
        
    @dataclass
    class Person:
        name: str
        age: int
        address: Address
        hobbies: List[str]
        
    people = [
        Person("Alice", 25, Address("北京", "朝阳区"), ["读书", "游泳"]),
        Person("Bob", 30, Address("上海", "浦东新区"), ["跑步", "音乐"]),
        Person("Charlie", 35, Address("深圳", "南山区"), ["编程", "旅游"]),
    ]
    
    for person in track(people, "处理 {name} ({age}岁) - 住址: {address.city} {address.street} - 爱好: {hobbies[0]}"):
        process(person)
    
    # 示例5：方法调用
    print("\n示例5：调用对象方法")
    strings = ["hello", "world", "python", "programming"]
    for s in track(strings, "处理字符串: {item} -> 大写: {item.upper()}"):
        process(s)
    
    # 示例6：复杂的嵌套类
    print("\n示例6：复杂嵌套类")
    
    class Project:
        def __init__(self, name, manager, team):
            self.name = name
            self.manager = manager
            self.team = team
            
        def get_team_size(self):
            return len(self.team)
    
    class Employee:
        def __init__(self, name, role):
            self.name = name
            self.role = role
    
    projects = [
        Project(
            "项目Alpha",
            Employee("张经理", "项目经理"),
            [Employee("开发1", "前端"), Employee("开发2", "后端")]
        ),
        Project(
            "项目Beta",
            Employee("李经理", "项目经理"),
            [Employee("开发3", "全栈"), Employee("开发4", "测试"), Employee("开发5", "运维")]
        ),
    ]
    
    for proj in track(projects, "项目: {name} - 经理: {manager.name} - 团队规模: {item.get_team_size()} - 首位成员: {team[0].name}"):
        process(proj)
    
    # 示例7：处理None值和默认值
    print("\n示例7：处理缺失值")
    partial_data = [
        {"name": "Alice", "email": "alice@example.com", "phone": "123456"},
        {"name": "Bob", "email": None, "phone": "789012"},
        {"name": "Charlie"},  # 缺少email和phone
    ]
    
    for item in track(partial_data, "用户: {name} - 邮箱: {email} - 电话: {phone}"):
        process(item)
    
    # 示例8：使用索引和总数
    print("\n示例8：显示进度信息")
    items = ["项目A", "项目B", "项目C", "项目D", "项目E"]
    for item in track(items, "[{i}/{t}] 正在处理: {item}"):
        process(item)
    
    # 示例9：并发处理复杂数据
    print("\n示例9：并发处理复杂数据")
    
    def worker(thread_id, data_list):
        for data in track(data_list, f"线程{thread_id} - 处理 {{item.id}}: {{item.data.value}}"):
            process(data)
    
    thread_data = [
        [{"id": f"T{tid}-{i}", "data": {"value": f"值{tid}-{i}"}} for i in range(5)]
        for tid in range(3)
    ]
    
    threads = []
    for tid, data in enumerate(thread_data):
        t = threading.Thread(target=worker, args=(tid, data))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    # 示例10：使用Progress类处理复杂数据
    print("\n示例10：使用Progress类")
    
    class Task:
        def __init__(self, id, name, status, priority):
            self.id = id
            self.name = name
            self.status = status
            self.priority = priority
            self.subtasks = []
            
        def add_subtask(self, subtask):
            self.subtasks.append(subtask)
            return self
            
        def get_info(self):
            return f"{self.name}({self.priority})"
    
    tasks = [
        Task(1, "任务A", "进行中", "高").add_subtask({"name": "子任务1", "done": False}),
        Task(2, "任务B", "待处理", "中").add_subtask({"name": "子任务2", "done": True}),
        Task(3, "任务C", "已完成", "低").add_subtask({"name": "子任务3", "done": False}),
    ]
    
    with Progress(tasks, "任务 {id}: {item.get_info()} - 状态: {status} - 子任务: {subtasks[0].name}", shared=True) as prog:
        for task in prog:
            process(task)
    
    # 示例11：动态更新描述
    print("\n示例11：动态更新描述")
    with Progress(shared=True) as prog:
        task_id = prog.add_task("初始化...", total=10)
        
        for i in range(10):
            # 创建动态数据
            current_data = {
                "step": i + 1,
                "status": "处理中" if i < 9 else "完成",
                "progress": f"{(i+1)*10}%"
            }
            
            # 格式化描述
            desc = format_description(
                "步骤 {step}/10 - 状态: {status} - 进度: {progress}",
                current_data
            )
            prog.progress.update(task_id, description=desc, advance=1)
            process(i)
    
    print("\n所有示例完成！")
