import heapq

n, k = map(int, input().split())
P = list(map(int, input().split()))
C = list(map(int, input().split()))

X = []
for i in range(n):
    X.append([P[i], C[i], i])
X.sort(key=lambda x: x[0])

coins = []
heapq.heapify(coins)
curr = 0
res = [0 for i in range(n)]

for i in range(k):
    heapq.heappush(coins, X[i][1])
    curr += X[i][1]
    res[X[i][2]] = curr

for j in range(k, n):
    res[X[j][2]] = X[j][1] + sum(coins)
    if len(coins) > 0:
        x = heapq.heappop(coins)
        if x < X[j][1]:
            heapq.heappush(coins, X[j][1])
        else:
            heapq.heappush(coins, x)

print(*res)