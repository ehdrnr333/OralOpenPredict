
photo = open("video_res.txt", 'r')
total_0 = 0
total_1 = 0
p_n_0 = 0
p_n_1 = 0
p_t_0 = 0
p_t_1 = 0

for i in range(0, 120):
    line = photo.readline()
    an = '0'
    if line[16] is '0':
        total_0 = total_0 + 1
        an = '0'
    else:
        total_1 = total_1 + 1
        an = '1'

    if line[8] is an:
        if line[8] is '0':
            p_t_0 = p_t_0 + 1
        else:
            p_t_1 = p_t_1 + 1

    if line[12] is an:
        if line[12] is '0':
            p_n_0 = p_n_0 + 1
        else:
            p_n_1 = p_n_1 + 1



print(p_t_0, p_t_1)
print(p_n_0, p_n_1)
print(total_0, total_1)


###############
photo = open("photo_res.txt", 'r')
total_0 = 0
total_1 = 0
p_n_0 = 0
p_n_1 = 0
p_t_0 = 0
p_t_1 = 0

for i in range(0, 53):
    line = photo.readline()
    an = '0'
    if line[19] is '0':
        total_0 = total_0 + 1
        an = '0'
    else:
        total_1 = total_1 + 1
        an = '1'

    if line[8] is an:
        if line[8] is '0':
            p_t_0 = p_t_0 + 1
        else:
            p_t_1 = p_t_1 + 1

    if line[12] is an:
        if line[12] is '0':
            p_n_0 = p_n_0 + 1
        else:
            p_n_1 = p_n_1 + 1



print(p_t_0, p_t_1)
print(p_n_0, p_n_1)
print(total_0, total_1)
