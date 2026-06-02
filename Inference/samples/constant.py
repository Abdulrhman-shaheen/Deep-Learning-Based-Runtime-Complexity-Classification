from math import *
a, vm = map(int, input().split())
l, d, vd = map(int, input().split())

if vm <= vd or sqrt(2 * a * d) <= vd:
    if vm ** 2 / (2 * a) >= l:
        ans = sqrt(2 * l / a)
    else:
        ans = vm / a + (l - vm ** 2 / (2 * a)) / vm
else:
    s1 = (vm ** 2 - vd ** 2) / (2 * a)
    if s1 >= l - d:
        ans = (sqrt(4 * vd ** 2 + 8 * a * (l - d)) - 2 * vd) / (2 * a)
    else:
        ans = (vm - vd) / a + (l - d - s1) / vm
    v1 = sqrt((2 * a * d + vd ** 2) / 2)
    if v1 <= vm:
        ans = ans + v1 / a + (v1 - vd) / a
    else:
        s1 = d - (vm ** 2 - vd ** 2) / (2 * a) - vm ** 2 / (2 * a)
        ans = ans + vm / a + (vm - vd) / a + s1 / vm

print('%.12f' % ans)