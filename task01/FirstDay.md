# 创建对象

## 常量

np.nan 空值//任意两个空值不相等  
np.e 自然常数  
np.inf 无穷大  
np.pi 圆周率

## 数据类型

| 类型                               |  备注 |           说明           |
| :--------------------------------- | ----: | :----------------------: |
| bool\_ = bool8                     |  8 位 |        布尔类型,         |
| int8 = byte                        |  8 位 |           整型           |
| int16 = short                      | 16 位 |           整型           |
| int32 = intc                       | 32 位 |           整型           |
| int\_ = int64 = long = int0 = intp | 64 位 |           整型           |
| uint8 = ubyte                      |  8 位 |        无符号整型        |
| uint16 = ushort                    | 16 位 |        无符号整型        |
| uint32 = uintc                     | 32 位 |        无符号整型        |
| uint64 = uintp = uint0 = uint      | 64 位 |        无符号整型        |
| float16 = half                     | 16 位 |          浮点型          |
| float32 = single                   | 32 位 |          浮点型          |
| float\_ = float64 = double         | 64 位 |          浮点型          |
| str*= unicode* = str0 = unicode    |       |      Unicode 字符串      |
| datetime64                         |       |       日期时间类型       |
| timedelta64                        |       | 表示两个时间之间的间隔\n |

## 创建数据类型

numpy 的数值类型实际上是 dtype 对象的实例。

```python
class dtype(object):
    def __init__(self, obj, align=False, copy=False):
        pass
```

每个内建类型都有一个唯一定义它的字符代码，如下：

| 类型 |                   备注 |           说明            |
| :--- | ---------------------: | :-----------------------: |
| 字符 |              "对应类型 |           备注            |
| "b"  |                boolean |           'b1'            |
| "i"  |         signed integer |  'i1' ,'i2' ,'i4' ,'i8'"  |
| "u"  |       unsigned integer |  'u1' ,'u2', 'u4' ,'u8'"  |
| "f"  |         floating-point |     'f2' ,'f4' ,'f8'      |
| "c"  | complex floating-point |                           |
| "m"  |            timedelta64 |  表示两个时间之间的间隔   |
| "M"  |             datetime64 |       日期时间类型        |
| "O"  |                 object |                           |
| "S"  |          (byte-)string | S3 表示长度为 3 的字符串" |
| "U"  |                Unicode |      Unicode 字符串"      |
| "V"  |                  void" |

## 日期时间和时间增量

在 numpy 中，我们很方便的将字符串转换成时间日期类型 datetime64（datetime 已被 python 包含的日期时间库所占用）。

datatime64 是带单位的日期时间类型，其单位如下：
| 日期单位 | 代码含义 | 时间单位 | 代码含义 |
| :------: | :------: | :------: | :------: |
| Y | 年 | h | 小时 |
| M | 月 | m | 分钟 |
| W | 周 | s | 秒 |
| D | 天 | ms | 毫秒 |
| - | - | us | 微秒 |
| - | - | ns | 纳秒 |
| - | - | ps | 皮秒 |
| - | - | fs | 飞秒 |
| - | - | as | 阿托秒 |

**[例]从字符串创建 datetime64 类型时，默认情况下，numpy 会根据字符串自动选择对应的单位**.

```python
import numpy as np
a = np.datetime64('2020-03-08 20:00:05')
print(a, a.dtype)  # 2020-03-08T20:00:05 datetime64[s]
```

**[例]从字符串创建 datetime64 数组时，如果单位不统一，则一律转化成其中最小的单位。**

```python
import numpy as np

a = np.array(['2020-03', '2020-03-08', '2020-03-08 20:00'], dtype='datetime64')
print(a, a.dtype)
# ['2020-03-01T00:00' '2020-03-08T00:00' '2020-03-08T20:00'] datetime64[m]
```

**[例]使用 arange()创建 datetime64 数组，用于生成日期范围。**  
生成器的步长与最小单位一致

```python
a = np.arange('2020-05', '2020-12', dtype=np.datetime64)
print(a)
# ['2020-05' '2020-06' '2020-07' '2020-08' '2020-09' '2020-10' '2020-11']
print(a.dtype)  # datetime64[M]
```

**[例]timedelta64 表示两个 datetime64 之间的差。timedelta64 也是带单位的，并且和相减运算中的两个 datetime64 中的较小的单位保持一致。**

```python
import numpy as np

a = np.datetime64('2020-03-08') - np.datetime64('2020-03-07')
b = np.datetime64('2020-03-08') - np.datetime64('202-03-07 08:00')
c = np.datetime64('2020-03-08') - np.datetime64('2020-03-07 23:00', 'D')

print(a, a.dtype)  # 1 days timedelta64[D]
print(b, b.dtype)  # 956178240 minutes timedelta64[m]
print(c, c.dtype)  # 1 days timedelta64[D]
```

**注意**生成 timedelta64 时，要注意年（'Y'）和月（'M'）这两个单位无法和其它单位进行运算（一年有几天,一个月有几个小时,这些都是不确定的）。
**[例]numpy.datetime64 与 Python 自带的 datetime.datetime 相互转换**

```python
import numpy as np
import datetime
dt = datetime.datetime(year=2020, month=6, day=1, hour=20, minute=5, second=30)
dt64 = np.datetime64(dt, 's')
print(dt64, dt64.dtype)
# 2020-06-01T20:05:30 datetime64[s]
dt2 = dt64.astype(datetime.datetime)
print(dt2, type(dt2))
# 2020-06-01 20:05:30 <class 'datetime.datetime'>
```

**[例题]为了允许在只有一周中某些日子有效的上下文中使用日期时间，NumPy 包含一组“busday”（工作日）功能。**

```python
numpy.busday_offset(dates, offsets, roll='raise', weekmask='1111100', holidays=None, busdaycal=None, out=None)
```

其中工作日默认周一到周五，周六周日休息。（所以我小时候，老师说外国人是周二到周六工作；周日，周一休息是骗我的喽？）  
**[例]自定义周掩码值，即指定一周中哪些星期是工作日。**

```python
import numpy as np

# 2020-07-10 星期五
a = np.is_busday('2020-07-10', weekmask=[1, 1, 1, 1, 1, 0, 0])
b = np.is_busday('2020-07-10', weekmask=[1, 1, 1, 1, 0, 0, 1])
print(a)  # True
print(b)  # False
```

更多涉及日期的操作查看官方文档，太无聊琐碎了，随用随查吧。

## 数组的创建

**Numpy 的核心操作**
numpy 提供的最重要的数据结构是 ndarray，它是 python 中 list 的扩展。

### 创建数组

[1]依据现有数据来创建 ndarray

```python
import numpy as np

# 创建一维数组
a = np.array([0, 1, 2, 3, 4])
b = np.array((0, 1, 2, 3, 4))
print(a, type(a))
# [0 1 2 3 4] <class 'numpy.ndarray'>
print(b, type(b))
# [0 1 2 3 4] <class 'numpy.ndarray'>

# 创建二维数组
c = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
print(c, type(c))
# [[11 12 13 14 15]
#  [16 17 18 19 20]
#  [21 22 23 24 25]
#  [26 27 28 29 30]
#  [31 32 33 34 35]] <class 'numpy.ndarray'>

# 创建三维数组
d = np.array([[(1.5, 2, 3), (4, 5, 6)],
              [(3, 2, 1), (4, 5, 6)]])
```

[2]通过 asarray()函数进行创建  
**array**和**asarray**都可以将数据结构转换成 ndarray 类型.但是主要区别就是当数据源是 ndarray 时，array 仍会 copy 出一个副本，占用新的内存，但 asarray 不会。

```python
【例】array()和asarray()都可以将结构数据转化为 ndarray

import numpy as np

x = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
y = np.array(x)
z = np.asarray(x)
x[1][2] = 2
print(x,type(x))
# [[1, 1, 1], [1, 1, 2], [1, 1, 1]] <class 'list'>

print(y,type(y))
# [[1 1 1]
#  [1 1 1]
#  [1 1 1]] <class 'numpy.ndarray'>

print(z,type(z))
# [[1 1 1]
#  [1 1 1]
#  [1 1 1]] <class 'numpy.ndarray'>
```

[3]通过 fromfunction()函数进行创建  
给函数绘图的时候可能会用到 fromfunction()，该函数可从函数中创建数组。

```python
def fromfunction(function, shape, **kwargs):
```

[例]通过在每个坐标上执行一个函数来构造数组。

```python
import numpy as np

def f(x, y):
    return 10 * x + y

x = np.fromfunction(f, (5, 4), dtype=int)
print(x)
# [[ 0  1  2  3]
#  [10 11 12 13]
#  [20 21 22 23]
#  [30 31 32 33]
#  [40 41 42 43]]

x = np.fromfunction(lambda i, j: i == j, (3, 3), dtype=int)
print(x)
# [[ True False False]
#  [False  True False]
#  [False False  True]]
x = np.fromfunction(lambda i, j: i + j, (3, 3), dtype=int)
print(x)
# [[0 1 2]
#  [1 2 3]
#  [2 3 4]]
```

### 依据 ones 和 zeros 填充方式

在机器学习任务中经常做的一件事就是初始化参数，需要用常数值或者随机值来创建一个固定大小的矩阵。  
**[1]数组**

```python
zeros()函数：返回给定形状和类型的零数组。
zeros_like()函数：返回与给定数组形状和类型相同的零数组。
def zeros(shape, dtype=None, order='C'):
def zeros_like(a, dtype=None, order='K', subok=True, shape=None):
【例】

import numpy as np
# 向量
x = np.zeros(5)
print(x)  # [0. 0. 0. 0. 0.]
# 指定大小
x = np.zeros([2, 3])
print(x)
# [[0. 0. 0.]
#  [0. 0. 0.]]

x = np.array([[1, 2, 3], [4, 5, 6]])
# 与其他矩阵维度相同的0矩阵
y = np.zeros_like(x)
print(y)
# [[0 0 0]
#  [0 0 0]]
```

[2]1 数组
ones()函数：返回给定形状和类型的 1 数组。

```python
def ones(shape, dtype=None, order='C'):
```

ones_like()函数：返回与给定数组形状和类型相同的 1 数组.

```python
def ones_like(a, dtype=None, order='K', subok=True, shape=None):
```

```python
import numpy as np

x = np.ones(5)
print(x)  # [1. 1. 1. 1. 1.]
x = np.ones([2, 3])
print(x)
# [[1. 1. 1.]
#  [1. 1. 1.]]
x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.ones_like(x)
print(y)
# [[1 1 1]
#  [1 1 1]]
```

[3]空数组
empty()函数：返回一个空数组，**数组元素为随机数。**

```python
def empty(shape, dtype=None, order='C'):
```

empty_like 函数：返回与给定数组具有相同形状和类型的新数组。

```python
def empty_like(prototype, dtype=None, order='K', subok=True, shape=None):
```

【例】

```python
import numpy as np
x = np.empty(5)
print(x)
# [1.95821574e-306 1.60219035e-306 1.37961506e-306
#  9.34609790e-307 1.24610383e-306]

x = np.empty((3, 2))
print(x)
# [[1.60220393e-306 9.34587382e-307]
#  [8.45599367e-307 7.56598449e-307]
#  [1.33509389e-306 3.59412896e-317]]

x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.empty_like(x)
print(y)
# [[  7209029   6422625   6619244]
#  [      100 707539280       504]]
```

**[4]单位数组**  
eye()函数：返回一个对角线上为 1，其它地方为零的单位数组。

```python
def eye(N, M=None, k=0, dtype=float, order='C'):
```

identity()函数：只能返回一个方阵。

```python
def identity(n, dtype=None):
```

【例】

```python
import numpy as np

x = np.eye(4)
print(x)
# [[1. 0. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [0. 0. 0. 1.]]

x = np.eye(2, 3)
print(x)
# [[1. 0. 0.]
#  [0. 1. 0.]]

x = np.identity(4)
print(x)
# [[1. 0. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [0. 0. 0. 1.]]
```

**案例：[（深度学习中的高级用法，将数组转成 one-hot 形式）](https://blog.csdn.net/m0_37393514/article/details/81455915)**

```python
import numpy as np

labels=np.array([[1],[2],[0],[1]])
print("labels的大小：",labels.shape,"\n")

#因为我们的类别是从0-2，所以这里是3个类
a=np.eye(3)[1]
print("如果对应的类别号是1，那么转成one-hot的形式",a,"\n")

a=np.eye(3)[2]
print("如果对应的类别号是2，那么转成one-hot的形式",a,"\n")

a=np.eye(3)[1,0]
print("1转成one-hot的数组的第一个数字是：",a,"\n")

#这里和上面的结果的区别，注意!!!
a=np.eye(3)[[1,2,0,1]]
print("如果对应的类别号是1,2,0,1，那么转成one-hot的形式\n",a)

res=np.eye(3)[labels.reshape(-1)]
print("labels转成one-hot形式的结果：\n",res,"\n")
print("labels转化成one-hot后的大小：",res.shape)
```

结果：

```python
labels的大小： (4, 1)

如果对应的类别号是1，那么转成one-hot的形式 [0. 1. 0.]

如果对应的类别号是2，那么转成one-hot的形式 [0. 0. 1.]

1转成one-hot的数组的第一个数字是： 0.0

如果对应的类别号是1,2,0,1，那么转成one-hot的形式
 [[0. 1. 0.]
 [0. 0. 1.]
 [1. 0. 0.]
 [0. 1. 0.]]
labels转成one-hot形式的结果：
 [[0. 1. 0.]
 [0. 0. 1.]
 [1. 0. 0.]
 [0. 1. 0.]]

labels转化成one-hot后的大小： (4, 3)
```

**[5]对角数组**  
diag()函数：提取对角线或构造对角数组。

```python
def diag(v, k=0):
```

【例】

```python
import numpy as np

x = np.arange(9).reshape((3, 3))
print(x)
# [[0 1 2]
#  [3 4 5]
#  [6 7 8]]
print(np.diag(x))  # [0 4 8]
print(np.diag(x, k=1))  # [1 5]
print(np.diag(x, k=-1))  # [3 7]

v = [1, 3, 5, 7]
x = np.diag(v)
print(x)
# [[1 0 0 0]
#  [0 3 0 0]
#  [0 0 5 0]
#  [0 0 0 7]]
```

**[6]常数数组**  
full()函数：返回一个常数数组。

```python
def full(shape, fill_value, dtype=None, order='C'):
```

full_like()函数：返回与给定数组具有相同形状和类型的常数数组。

```python
def full_like(a, fill_value, dtype=None, order='K', subok=True, shape=None):
```

【例】

```python
import numpy as np

x = np.full((2,), 7)
print(x)
# [7 7]

x = np.full(2, 7)
print(x)
# [7 7]

x = np.full((2, 7), 7)
print(x)
# [[7 7 7 7 7 7 7]
#  [7 7 7 7 7 7 7]]

x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.full_like(x, 7)
print(y)
# [[7 7 7]
#  [7 7 7]]
```

### 利用数值范围创建 ndarry

arange()函数：返回给定间隔内的均匀间隔的值。

```python
def arange([start,] stop[, step,], dtype=None):
```

linspace()函数：返回指定间隔内的等间隔数字。

```python
def linspace(start, stop, num=50, endpoint=True, retstep=False,
             dtype=None, axis=0):
```

logspace()函数：返回数以对数刻度均匀分布。

```python
def logspace(start, stop, num=50, endpoint=True, base=10.0,
             dtype=None, axis=0):
```

numpy.random.rand() 返回一个由[0,1)内的随机数组成的数组。

```python
def rand(d0, d1, ..., dn):
```

【例】

```python
import numpy as np

x = np.arange(5)
print(x)  # [0 1 2 3 4]

x = np.arange(3, 7, 2)
print(x)  # [3 5]

x = np.linspace(start=0, stop=2, num=9)
print(x)
# [0.   0.25 0.5  0.75 1.   1.25 1.5  1.75 2.  ]

x = np.logspace(0, 1, 5)
print(np.around(x, 2))
# [ 1.    1.78  3.16  5.62 10.  ]
                                    #np.around 返回四舍五入后的值，可指定精度。
                                   # around(a, decimals=0, out=None)
                                   # a 输入数组
                                   # decimals 要舍入的小数位数。 默认值为0。 如果为负，整数将四舍五入到小数点左侧的位置


x = np.linspace(start=0, stop=1, num=5)
x = [10 ** i for i in x]
print(np.around(x, 2))
# [ 1.    1.78  3.16  5.62 10.  ]

x = np.random.random(5)
print(x)
# [0.41768753 0.16315577 0.80167915 0.99690199 0.11812291]

x = np.random.random([2, 3])
print(x)
# [[0.41151858 0.93785153 0.57031309]
#  [0.13482333 0.20583516 0.45429181]]
```

### 4. 结构数组的创建

结构数组，首先需要定义结构，然后利用 np.array()来创建数组，其参数 dtype 为定义的结构。

[1]利用字典来定义结构  
【例】

```python
import numpy as np

personType = np.dtype({
    'names': ['name', 'age', 'weight'],
    'formats': ['U30', 'i8', 'f8']})

a = np.array([('Liming', 24, 63.9), ('Mike', 15, 67.), ('Jan', 34, 45.8)],
             dtype=personType)
print(a, type(a))
# [('Liming', 24, 63.9) ('Mike', 15, 67. ) ('Jan', 34, 45.8)]
# <class 'numpy.ndarray'>
```

[2]利用包含多个元组的列表来定义结构  
【例】

```python
import numpy as np

personType = np.dtype([('name', 'U30'), ('age', 'i8'), ('weight', 'f8')])
a = np.array([('Liming', 24, 63.9), ('Mike', 15, 67.), ('Jan', 34, 45.8)],
             dtype=personType)
print(a, type(a))
# [('Liming', 24, 63.9) ('Mike', 15, 67. ) ('Jan', 34, 45.8)]
# <class 'numpy.ndarray'>

# 结构数组的取值方式和一般数组差不多，可以通过下标取得元素：
print(a[0])
# ('Liming', 24, 63.9)

print(a[-2:])
# [('Mike', 15, 67. ) ('Jan', 34, 45.8)]

# 我们可以使用字段名作为下标获取对应的值
print(a['name'])
# ['Liming' 'Mike' 'Jan']
print(a['age'])
# [24 15 34]
print(a['weight'])
# [63.9 67.  45.8]
```

数组的属性
在使用 numpy 时，你会想知道数组的某些信息。很幸运，在这个包里边包含了很多便捷的方法，可以给你想要的信息。

**numpy.ndarray.ndim**用于返回数组的维数（轴的个数）也称为秩，一维数组的秩为 1，二维数组的秩为 2，以此类推。  
**numpy.ndarray.shape**表示数组的维度，返回一个元组，这个元组的长度就是维度的数目，即 ndim 属性(秩)。  
**numpy.ndarray.size**数组中所有元素的总量，相当于数组的 shape 中所有元素的乘积，例如矩阵的元素总量为行与列的乘积。  
**numpy.ndarray.dtype**: ndarray 对象的元素类型。  
**numpy.ndarray.itemsize**:数组中每个元素的字节数的大小

```python
class ndarray(object):
    shape = property(lambda self: object(), lambda self, v: None, lambda self: None)
    dtype = property(lambda self: object(), lambda self, v: None, lambda self: None)
    size = property(lambda self: object(), lambda self, v: None, lambda self: None)
    ndim = property(lambda self: object(), lambda self, v: None, lambda self: None)
    itemsize = property(lambda self: object(), lambda self, v: None, lambda self: None)
```

【例】

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(a.shape)  # (5,)
print(a.dtype)  # int32
print(a.size)  # 5
print(a.ndim)  # 1
print(a.itemsize)  # 4

b = np.array([[1, 2, 3], [4, 5, 6.0]])
print(b.shape)  # (2, 3)
print(b.dtype)  # float64
print(b.size)  # 6
print(b.ndim)  # 2
print(b.itemsize)  # 8
```

在 ndarray 中所有元素必须是同一类型，否则会自动向下转换，int->float->str。

【例】

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(a)  # [1 2 3 4 5]
b = np.array([1, 2, 3, 4, '5'])
print(b)  # ['1' '2' '3' '4' '5']
c = np.array([1, 2, 3, 4, 5.0])
print(c)  # [1. 2. 3. 4. 5.]
```
