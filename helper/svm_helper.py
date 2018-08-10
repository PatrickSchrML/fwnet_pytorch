import numpy as np


def linear_pred(sv_w, sv_x, sv_y, x, num_examples):
    sv_y = np.reshape(sv_y, [num_examples])
    sv_w = np.reshape(sv_w, [num_examples])

    # Create Prediction Kernel
    # Linear prediction kernel
    pred_kernel = np.matmul(sv_x, np.transpose(x))

    # ---
    sv_wy = np.multiply(sv_y, sv_w)
    prediction_output = np.matmul(sv_wy, pred_kernel)
    prediction = np.sign(prediction_output - np.mean(prediction_output))
    #prediction = np.sign(prediction_output)

    return prediction


def rbf_pred(sv_w, sv_x, sv_y, x, gamma, num_examples):
    sv_y = np.reshape(sv_y, [num_examples])
    sv_w = np.reshape(sv_w, [num_examples])

    # Create Prediction Kernel
    # Linear prediction kernel
    # pred_kernel = tf.matmul(x_data, tf.transpose(prediction_grid))

    # Gaussian (RBF) prediction kernel
    rA = np.reshape(np.sum(np.square(sv_x), 1), [-1, 1])
    rB = np.reshape(np.sum(np.square(x), 1), [-1, 1])
    pred_sq_dist = np.add(np.subtract(rA, np.multiply(2., np.matmul(sv_x, np.transpose(x)))),
                          np.transpose(rB))
    pred_kernel = np.exp(np.multiply(gamma, np.abs(pred_sq_dist)))

    # ---
    sv_wy = np.multiply(sv_y, sv_w)
    prediction_output = np.matmul(sv_wy, pred_kernel)
    prediction = np.sign(prediction_output - np.mean(prediction_output))
    #prediction = np.sign(prediction_output)

    return prediction


def get_acc(sv_w, sv_x, sv_y, x, gt, c, gamma, num_examples, step, print_acc=True, kernel_type="linear"):
    if kernel_type == "linear":
        prediction = linear_pred(sv_w, sv_x, sv_y, x, num_examples)
    else:
        prediction = rbf_pred(sv_w, sv_x, sv_y, x, gamma, num_examples)
    #print(np.squeeze(prediction))
    #print(np.squeeze(gt))
    accuracy = np.mean(np.equal(np.squeeze(prediction), np.squeeze(gt)))

    if print_acc:
        print('Step %d: Accuracy %.9f' % (step, accuracy))

    return accuracy


def get_loss_np(alpha, k):
    # TODO tf.multiply(0.5)
    return np.matmul(np.matmul(np.transpose(alpha), k), alpha)


def preprocess_svm_data(data):
    max_value_axis = np.amax(np.abs(data), axis=0)
    data_preprocessed = data / max_value_axis
    return data_preprocessed

########################################################################################################################
