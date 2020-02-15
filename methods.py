import numpy as np
import matplotlib.pyplot as plt

# Complementary Functions

# Weight vector initialization
def weights_init(m, key):
    # key = True => zero initialization
    # key = False => Random initialization
    if key:
        w = np.zeros((m, 1))
    else:
        w = np.random.randi(m, 1)

    return w


###############################################################################

# Text pre-processing

def text_prep(text):
    punctuations_list = '!-[]{};\,<>/"?#$%^&*_~+' + '\n\n'
    output_text = ""
    text = text.strip()

    for ch in text:
        if ch not in punctuations_list:
            output_text = output_text + ch

    output_text = output_text.lower()
    prep_text = output_text.split(" ")
    prep_text = list(filter(None, prep_text))

    return prep_text


###############################################################################

# Most-frequent word count features for the whole Training set
def Feature_Matrix(dataset, no_txt_features):
    offensive_words_list = ['shit', 'fuck', 'fucking', 'bitch', 'damn', 'sex', 'ass', 'hell',
                            'hot', 'dick', 'shitty', 'fucked', 'asshole', 'bullshit', 'gay', 'porn', 'crap', 'sucks']

    positive_sentiments_list = ['like', 'really', 'good', 'please', 'love', 'pretty', 'best',
                                'better', 'great', 'movie', 'happy', 'watching', 'nice', 'fun',
                                'thanks', 'thank', 'funny', 'cool', 'thankfully', 'super', 'enjoy',
                                'awesome', 'wow', 'amazing', 'interesting', 'loved’, ‘liked', 'perfect',
                                'fan', 'glad', 'haha', 'fans', 'hilarious', 'popular', 'fair', 'special',
                                'beautiful', ':d', ':)', '=d']

    negative_sentiments_list = ['bad', 'hate', 'wrong', 'lost', 'damn', 'hell', 'sorry', 'dead', 'weird', 'shitty',
                                'worst', 'terrible', 'worse', 'sad', 'seeing', 'die', 'death', 'died', 'kill', 'poor',
                                'breaking', 'horrible', ':(', '=(']

    N = len(dataset)
    All_comments = [None] * N
    Words_Dict = {}
    bias = np.ones((N, 1))
    controversiality_vec = np.zeros((N, 1))
    is_root_vec = np.zeros((N, 1))
    children_vec = np.zeros((N, 1))
    Y = np.zeros((N, 1))
    X_words_count = np.zeros((N, no_txt_features))
    offensive_count = np.zeros((N, 1))
    http_count = np.zeros((N, 1))
    positive_sentiments = np.zeros((N, 1))
    negative_sentiments = np.zeros((N, 1))

    for i in range(N):
        for key, val in dataset[i].items():
            if key == 'text':
                preprocessed_text = text_prep(val)
                All_comments[i] = preprocessed_text
                for x in preprocessed_text:
                    if x not in Words_Dict.keys():
                        Words_Dict[x] = 0
                    Words_Dict[x] += 1

            elif key == 'is_root':
                if val:
                    is_root_vec[i] = 1
                else:
                    is_root_vec[i] = 0

            elif key == 'controversiality':
                controversiality_vec[i] = val

            elif key == 'children':
                children_vec[i] = val

            elif key == 'popularity_score':
                Y[i] = val

    # Now we need to sort out from most-frequent to least-frequent words in dictionary to obtain the first N words
    if no_txt_features == 0:

        X1 = np.append(children_vec, controversiality_vec, axis=1)
        X1 = np.append(X1, is_root_vec, axis=1)
        X = np.append(X1, bias, axis=1)

    else:
        Words_Dict_Sorted = sorted(Words_Dict.items(), key=lambda t: t[1], reverse=True)
        Most_Freq_Words_Dict = dict(list(Words_Dict_Sorted[:no_txt_features]))
        Most_Freq_Words = list(Most_Freq_Words_Dict.keys())

        # Now we need to count the frequency of most frequent words in each comment throughout the whole dataset

        # Most-Frequent words
        for i in range(N):
            for j in range(no_txt_features):
                X_words_count[i, j] = All_comments[i].count(Most_Freq_Words[j])

                # Offensive and HTTP-containing comments

        for i in range(N):
            for x in All_comments[i]:
                if x.find('http') or x.find('www.') != -1:
                    http_count[i] = 1

                for l in range(len(positive_sentiments_list)):
                    if x.find(positive_sentiments_list[l]) != -1:
                        positive_sentiments[i] += 1

                for l in range(len(negative_sentiments_list)):
                    if x.find(negative_sentiments_list[l]) != -1:
                        negative_sentiments[i] += 1

            for j in range(len(offensive_words_list)):
                offensive_count[i] += All_comments[i].count(offensive_words_list[j])

        for j in range(len(offensive_words_list)):
            if offensive_count[i] > 0:
                offensive_count[i] = 1

        # Positive or negative sentiments

        X1 = np.append(X_words_count, controversiality_vec, axis=1)
        X1 = np.append(X1, is_root_vec, axis=1)
        X1 = np.append(X1, children_vec, axis=1)
        X1 = np.append(X1, offensive_count, axis=1)
        X1 = np.append(X1, http_count, axis=1)
        X1 = np.append(X1, positive_sentiments, axis=1)
        X1 = np.append(X1, negative_sentiments, axis=1)

        # Data Rescaling

        for i in range(X1.shape[1]):
            mean = np.mean(X1[:, i])
            var = np.var(X1[:, i])
            X1[:, i] = (1 / np.sqrt(var)) * (X1[:, i] - mean)

        X = np.append(X1, bias, axis=1)

    return X, Y, Most_Freq_Words_Dict


###############################################################################

# Visualisations
def lineplot(x_data, y_data, x_label="", y_label="", title="", gcolor=""):
    # Create the plot object
    plt.figure()
    # Plot the best fit line, set the linewidth (lw), color and
    # transparency (alpha) of the line
    plt.plot(x_data, y_data, lw=2, color=gcolor, alpha=1)
    # Label the axes and provide a title
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


###############################################################################

# MSE
def Mean_Square_Error(X, Y, w_hat):
    prediction = np.dot(X, w_hat)
    abs_err = np.subtract(Y, prediction)
    squared_err = np.square(abs_err)
    MSE = (1 / X.shape[0]) * np.sum(squared_err)
    return MSE


###############################################################################

# Gradient Descent
def Gradient_Descent(X, Y, W0, B, E, eps):
    eta0 = E
    beta = B
    epsilon = eps
    w0 = W0
    mse = np.zeros(100000000)  # some big number for number of epochs
    X_T = np.dot(X.T, X)
    X_Y = np.dot(X.T, Y)
    alpha = eta0 / (1 + beta)
    w_gd = w0 - 2 * alpha * (np.subtract(np.dot(X_T, w0), X_Y))
    diff = np.linalg.norm(np.subtract(w_gd, w0))
    epoch = 0
    mse[epoch] = Mean_Square_Error(X, Y, w_gd)

    while diff > epsilon:
        w0 = w_gd
        alpha = eta0 / (1 + beta * (epoch + 1))
        w_gd = w0 - 2 * alpha * np.subtract((X_T).dot(w0), X_Y)
        diff = np.linalg.norm(np.subtract(w_gd, w0))
        epoch += 1
        mse[epoch] = Mean_Square_Error(X, Y, w_gd)

    MSE = np.delete(mse, np.s_[epoch + 1: len(mse) + 1])  # Removing zero-valued MSE at the end

    return w_gd, MSE


###############################################################################

# Least-Square estimation

def Least_Squares_Estimation(X, Y):
    X_T = (X.T).dot(X)
    X_T_inv = np.linalg.inv(X_T)
    X_Y = (X.T).dot(Y)
    w_hat = np.dot(X_T_inv, X_Y)

    return w_hat

