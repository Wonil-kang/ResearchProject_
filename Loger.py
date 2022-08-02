import fileIO


def log(algorithm, activation_functions, data_type, number_of_units, window_size, mse, accuracy, result_file_path):

    title = algorithm + "," + activation_functions + "," + str(data_type) + "," + str(number_of_units) + "," + str(
        window_size)

    statistics_result = str(mse) + "," + str(accuracy)
    final_result = title + "," + statistics_result + "," + result_file_path

    print(final_result)

    fileIO.logResult(final_result)



def log(str):

    print(str)
    fileIO.logResult(str)