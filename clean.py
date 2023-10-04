# Membaca isi file teks dan menghapus duplikat
input_file = 'links.txt'
output_file = 'links_cleaned.txt'

with open(input_file, 'r') as file:
    lines = file.readlines()

# Menghilangkan duplikat
unique_lines = list(set(lines))

# Menyimpan hasilnya ke dalam file lain
with open(output_file, 'w') as file:
    file.writelines(unique_lines)

print("Duplikat telah dihapus dan hasil disimpan di", output_file)
