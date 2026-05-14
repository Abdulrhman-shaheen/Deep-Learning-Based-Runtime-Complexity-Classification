# ######### first code #########

n, m = map(int, input().split())

blocked = [int(input()) for _ in range(n)]
blocked.append(10**9)

selected = []

for _ in range(m):
    a, b, c = map(int, input().split())
    if a == 1:
        selected.append(b)

blocked.sort()
selected.sort()

total = len(selected)
answer = total
pointer = 0

for i in range(n + 1):
    while pointer < total and selected[pointer] < blocked[i]:
        pointer += 1

    answer = min(answer, total - pointer + i)

    if pointer == total:
        break

print(answer)




########### second code #########


# import sys

# n, m = map(int, input().split())
# grid = [list(input()) for _ in range(n)]

# up = [[0] * m for _ in range(n)]
# down = [[0] * m for _ in range(n)]
# left = [[0] * m for _ in range(n)]
# right = [[0] * m for _ in range(n)]

# for col in range(m):
#     count = 0
#     for row in range(n):
#         count = count + 1 if grid[row][col] == '*' else 0
#         up[row][col] = count

#     count = 0
#     for row in range(n - 1, -1, -1):
#         count = count + 1 if grid[row][col] == '*' else 0
#         down[row][col] = count

# for row in range(n):
#     count = 0
#     for col in range(m):
#         count = count + 1 if grid[row][col] == '*' else 0
#         left[row][col] = count

#     count = 0
#     for col in range(m - 1, -1, -1):
#         count = count + 1 if grid[row][col] == '*' else 0
#         right[row][col] = count

# stars = []
# vertical_marks = [[0] * m for _ in range(n)]
# horizontal_marks = [[0] * m for _ in range(n)]

# for row in range(n):
#     for col in range(m):
#         size = min(
#             up[row][col],
#             down[row][col],
#             left[row][col],
#             right[row][col]
#         ) - 1

#         if size > 0:
#             stars.append((row + 1, col + 1, size))

#             vertical_marks[row - size][col] += 1
#             vertical_marks[row + size][col] -= 1

#             horizontal_marks[row][col - size] += 1
#             horizontal_marks[row][col + size] -= 1

# reconstructed = [['.' for _ in range(m)] for _ in range(n)]

# for row in range(n):
#     active = 0
#     for col in range(m):
#         active += horizontal_marks[row][col]
#         if active != 0 or horizontal_marks[row][col] != 0:
#             reconstructed[row][col] = '*'

# for col in range(m):
#     active = 0
#     for row in range(n):
#         active += vertical_marks[row][col]
#         if active != 0 or vertical_marks[row][col] != 0:
#             reconstructed[row][col] = '*'

# if reconstructed != grid:
#     print(-1)
#     sys.exit()

# print(len(stars))
# for star in stars:
#     print(*star)



x = int(input())
for i in range(x):
    for j in range(x):
        if i == j or i + j == x - 1:
            print("*", end="")
        else:
            print(" ", end="")
