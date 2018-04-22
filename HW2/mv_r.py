def get_rating(ratings):
    if not ratings:
        return 0
    dp = [0] * (len(ratings) + 1)
    dp[-1] = 0
    dp[1] = ratings[0]
    for i in range(len(ratings)-2,0,-1):
        print(i)
        dp[i] = ratings[i] + max(dp[i + 1], dp[i + 2])
    return dp




# max([ratings[0] + get_rating(ratings[1:]),
                # ratings[1] + get_rating(ratings[2:])])
print get_rating([9, -1, -3, 4, 5])  # 17
print get_rating([-1, -3, -2])
print get_rating([5, 9, -1, -3, 4, 5])
