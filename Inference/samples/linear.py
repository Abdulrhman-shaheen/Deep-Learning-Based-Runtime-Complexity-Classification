n = int(input())
arr = list(map(int, input().split()))

total = 0
for x in arr:
    total += x

print(total / n)