
anchor_list = [[10,14], [23,27], [37,58], [81,82], [135,169], [344,319]]

flt_anchor_list = list()
for x, y in anchor_list:
    x = round(x/416, 6)
    y = round(y/416, 6)
    flt_anchor_list.append([x, y])

print(flt_anchor_list)