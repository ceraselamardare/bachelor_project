import csv

list_headers = ['image_id', 'bald', 'black_hair', 'blond_hair', 'brown_hair', 'gray_hair']

first_row_read = True
total_img = 0
number_of_files = 1
with open(r'C:\Users\Cera\PycharmProjects\pythonProject\procesare\processed_celeb_faces.csv', mode='w') as csv_to_write:
    csv_writer = csv.writer(csv_to_write, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    with open(r'C:\Users\Cera\PycharmProjects\pythonProject\procesare\list_attr_celeba.csv') as csv_to_read:
        csv_reader = csv.reader(csv_to_read, delimiter=',')
        for row in csv_reader:
            if first_row_read:
                csv_writer.writerow(list_headers)
                first_row_read = False
            else:
                list_append = []
                if row[5] == '-1' and row[9] == '-1' and row[10] == '-1' and row[12] == '-1' and row[18] == '-1':
                    continue
                list_append.append(row[0])
                list_append.append(row[5] if row[5] == '1' else '0')
                list_append.append(row[9] if row[9] == '1' else '0')
                list_append.append(row[10] if row[10] == '1' else '0')
                list_append.append(row[12] if row[12] == '1' else '0')
                list_append.append(row[18] if row[18] == '1' else '0')

                s = 0
                for val in list_append[1:]:
                    s = s + int(val)
                if s == 1:
                    csv_writer.writerow(list_append)
                    total_img += 1

print(total_img)