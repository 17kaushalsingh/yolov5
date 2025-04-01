from utils.plots import plot_results

path = input("Enter the path to the training results.csv file relative to yolov5 directory: ")
plot_results(path)  # This will generate 'results.png' in the same directory as results.csv