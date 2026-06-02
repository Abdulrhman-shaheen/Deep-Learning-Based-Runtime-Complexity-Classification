n, target = map(int, input().split())
arr = list(map(int, input().split()))

lo, hi = 0, n - 1
result = -1

while lo <= hi:
    mid = (lo + hi) // 2
    if arr[mid] == target:
        result = mid
        break
    elif arr[mid] < target:
        lo = mid + 1
    else:
        hi = mid - 1

print(result)