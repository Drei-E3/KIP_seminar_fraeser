import subprocess
import sys

def main_menu():
    print("\nImage Processing and Modeling Application")
    print("1. Image Augmentation")
    print("2. Image Cropping")
    print("3. Image Grayscaling")
    print("4. Model Operations")
    print("5. Exit")
    choice = input("Enter your choice (1-5): ")
    return choice

def run_script(script, args):
    try:
        result = subprocess.run([sys.executable, script] + args, capture_output=True, text=True)
        print("Output:", result.stdout)
        if result.stderr:
            print("Error:", result.stderr)
    except Exception as e:
        print("An error occurred:", e)

def augmentation():

    in_folder = input("Enter the path to the input folder: ")
    out_folder = input("Enter the path to the output folder: ")
    run_script("utils/augumentation.py", ["--in_folder", in_folder, "--out_folder", out_folder])

def cropping():

    in_folder = input("Enter the path to the input folder: ")
    out_folder = input("Enter the path to the output folder: ")
    width = input("Enter the crop width: ")
    height = input("Enter the crop height: ")
    run_script("utils/cropping.py", ["--in_folder", in_folder, "--out_folder", out_folder, "--crop_width", width, "--crop_height", height])

def grayscaling():

    in_folder = input("Enter the path to the input folder: ")
    out_folder = input("Enter the path to the output folder: ")
    run_script("utils/grayscaling.py", ["--in_folder", in_folder, "--out_folder", out_folder])

def modeling():

    action = input("Choose action (load, train, evaluate): ")
    image_path = input("Enter the path to the image folder: ")
    mask_path = input("Enter the path to the mask folder: ")
    model_path = input("Enter the path to the model file (if applicable): ")
    run_script("utils/modeling.py", ["--action", action, 
                                     "--image_path", image_path, 
                                     "--mask_path",mask_path,
                                     "--model_path", model_path])

if __name__ == "__main__":
    while True:
        choice = main_menu()
        if choice == '1':
            augmentation()
        elif choice == '2':
            cropping()
        elif choice == '3':
            grayscaling()
        elif choice == '4':
            modeling()
        elif choice == '5':
            print("Exiting application.")
            break
        else:
            print("Invalid choice. Please select a valid option.")