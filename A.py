from main import replace_accented
from sklearn import svm
from sklearn import neighbors
import nltk

# don't change the window size
window_size = 10

# A.1
#what went wrong: the build_s is the words in all of the contexts, below in the vectorize you need to
# count the words in a particular context
def build_s(data):
    '''
    Compute the context vector for each lexelt
    :param data: dic with the following structure:
        {
			lexelt: [(instance_id, left_context, head, right_context, sense_id), ...],
			...
        }
    :return: dic s with the following structure:
        {
			lexelt: [w1,w2,w3, ...],
			...
        }

    '''
    s = {}
    for lexelt, value in data.iteritems():
        for iid, lc, h, rc, sid in value:
            set_of_words = []
            left_context = nltk.word_tokenize(lc)
            right_context = nltk.word_tokenize(rc)
            context = left_context[-window_size:] + right_context[:window_size]
            for w in context:
                if w not in set_of_words:
                    set_of_words.append(w)
            s[lexelt] = set_of_words
    return s


# A.1
def vectorize(data, s):
    # you find the set of words that appear in a window of k = 10 in all contexts (Sall), then you check for each
    # context how many times each word of the set occurs. That's the vector.
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :param s: list of words (features) for a given lexelt: [w1,w2,w3, ...]
    :return: vectors: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }

    '''
    vectors = {}
    labels = {}
    for lexelt in data:
        left_context = nltk.word_tokenize(lexelt[1])
        right_context = nltk.word_tokenize(lexelt[3])
        context = left_context[-window_size:] + right_context[:window_size]
        vectors[lexelt[0]] = [context.count(w) for w in s]
        labels[lexelt[0]] = lexelt[-1]

    return vectors, labels


# A.2
def classify(X_train, X_test, y_train):
    '''
    Train two classifiers on (X_train, and y_train) then predict X_test labels

    :param X_train: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param X_test: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }

    :return: svm_results: a list of tuples (instance_id, label) where labels are predicted by LinearSVC
             knn_results: a list of tuples (instance_id, label) where labels are predicted by KNeighborsClassifier
    '''

    svm_results = []
    knn_results = []

    svm_clf = svm.LinearSVC()
    knn_clf = neighbors.KNeighborsClassifier()

    # we need to map each vector to a sense_id. each is a in different dictionary (not ordered), both can be recovered
    # because they have the same instance_id as key. We put them in a list because lists are ordered.
    X_parameter = []
    y_parameter = []
    for instance_id, vec in X_train.iteritems():
        X_parameter.append(vec)
        y_parameter.append(y_train[instance_id])

    #to train the svm
    svm_clf.fit(X_parameter, y_parameter)

    # to train the knn
    knn_clf.fit(X_parameter, y_parameter)

    # the predict gives you a list, so since we do not want a list in the tuple (instance_id, label), we just ask for
    # the first (and only) element in the list returned by predict
    #
    for instance_id, vec in X_test.iteritems():
        svm_results.append((instance_id, svm_clf.predict(vec)[0]))
        knn_results.append((instance_id, knn_clf.predict(vec)[0]))

    return svm_results, knn_results

# A.3, A.4 output
def print_results(results ,output_file):
    '''

    :param results: A dictionary with key = lexelt and value = a list of tuples (instance_id, label)
    :param output_file: file to write output

    '''

    # implement your code here
    # don't forget to remove the accent of characters using main.replace_accented(input_str)
    # you should sort results on instance_id before printing
    with open(output_file, 'w') as output:
        for lexelt, value in results.iteritems():
            for instance in sorted(value, key=lambda x:x[0]):
                x = replace_accented(lexelt)
                y = replace_accented(instance[0])
                z = instance[1]
                output.write(x + ' ' + y + ' ' + z + '\n')


# run part A
def run(train, test, language, knn_file, svm_file):
    s = build_s(train)
    svm_results = {}
    knn_results = {}
    for lexelt in s:
        X_train, y_train = vectorize(train[lexelt], s[lexelt])
        X_test, _ = vectorize(test[lexelt], s[lexelt])
        svm_results[lexelt], knn_results[lexelt] = classify(X_train, X_test, y_train)

    print_results(svm_results, svm_file)
    print_results(knn_results, knn_file)



