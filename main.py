# main.py
from core.dataset_builder import generate_dataset_from_directory

def main():
    generate_dataset_from_directory('images/', output_csv='haralick_dataset.csv')
    print("Extração concluída e CSV salvo incrementalmente.")

if __name__ == '__main__':
    main()
