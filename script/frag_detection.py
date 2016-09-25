#! /usr/bin/env python

from sklearn.linear_model import LogisticRegression
from nltk.classify.scikitlearn import SklearnClassifier
import stanford_parse
from collections import Counter

DATA = {}

# Required files with each dataset:
# 1. file containing the sentences with the fragments flagged with '*' (nucle_frag_model.txt)
# 2. file containing only the sentences (nucle_frag.txt)
# 3. file with the lemma and pos of the words (nucle_frag_lpos.txt)
# 4. file with the parse trees produced by the Stanford Parser (nucle_frag_trees.txt)

DATA['nucle_frag_f'] = ['../data/nucle_frag_model.txt', '../data/nucle_frag.txt']

best_ratio = {
    'sva': [1, 1, 1, 1],
    'cs': [1, 1, 1, 1],
    'frag': [1, 1, 1, 1],
    'mp': [1, 1, 1, 1],
}

prob_list = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


FIND_HEAD_DICT = {'ADJP':(0,['NNS','QP','NN','ADVP','JJ','VBN','VBG','ADJP','JJR','NP','JJS','DT','FW','RBR','RBS','SBAR','RB']),'ADVP':(1,['RB','RBR','RBS','FW','ADVP','TO','CD','JJR','JJ','IN','NP','JJS','NN']),'CONJP':(1,['CC','RB','IN']),'FRAG':(1,[]),'INTJ':(0,[]),'LST':(1,['LS']),'NAC':(0,['NN','NNS','NNP','NNPS','NP','NAC','EX','CD','QP','PRP','VBG','JJ','JJS','JJR','ADJP','FW']),'PP':(1,['IN','TO','VBG','VBN','RP','FW']),'PRN':(0,[]),'PRT':(1,['RP']),'QP':(0,['$','IN','NNS','NN','JJ','RB','DT','CD','NCD','QP','JJR','JJS']),'RRC':(1,['VP','NP','ADVP','ADJP','PP']),'S':(0,['TO','IN','VP','S','SBAR','ADJP','UCP','NP']),'SBAR':(0,['WHNP','WHPP','WHADVP','WHADJP','IN','DT','S','SQ','SINV','SBAR','FRAG']),'SBARQ':(0,['SQ','S','SINV','SBARQ','FRAG']),'SINV':(0,['VBZ','VBD','VBP','VB','MD','VP','S','SINV','ADJP','NP']),'SQ':(0,['VBZ','VBD','VBP','VB','MD','VP','SQ']),'UCP':(1,[]),'VP':(0,['TO','VBD','VBN','MD','VBZ','VB','VBG','VBP','VP','ADJP','NN','NNS','NP']),'WHADJP':(0,['CC','WRB','JJ','ADJP']),'WHADVP':(1,['CC','WRB']),'WHNP':(0,['WDT','WP','WP$','WHADJP','WHPP','WHNP']),'WHPP':(1,['IN','TO','FW'])}

convert_noun = False
print_error = False

filter_threshold = 75

def encode_features(data, freq_required=1):
    features_count = Counter()
    for x in data:
        for y in x[0]:
            features_count[y] += 1
    features_set = [x for x in features_count.keys() if features_count[x]>=freq_required]
    print( "number of features: {}".format(len(features_set)) )
    return (features_set, encode_features_with_feature_set(data, features_set))

def encode_features_with_feature_set(data, features_set):
    all_features_results = []
    for x in data:
        features = {}
        for f in features_set:
            features[f] = 1 if f in x[0] else 0
        all_features_results.append((features, x[1]))
    return all_features_results

def train_maxent(training_data):
    print( "training..." )
    features_set, all_features_results = encode_features(training_data, filter_threshold)
    classifier = SklearnClassifier(LogisticRegression(C=1.0, class_weight='balanced'))
    classifier.train(all_features_results)
    return (features_set, classifier)

def precision_recall_accuracy(counts):
    t_count = counts[0]
    m_count = counts[1]
    c_count = counts[2]
    a_count = counts[3]
    all_count = counts[4]

    if t_count > 0:
        P = float(c_count)/t_count
    else:
        P = 0
    if m_count > 0:
        R = float(c_count)/m_count
    else:
        R = 0

    print( "# model: {}".format(m_count) )
    print( "# test: {}".format(t_count) )
    print( "# correct: {}".format(c_count) )
    print( "# match: {}".format(a_count) )
    print( "# all: {}".format(all_count) )
    print
    print( "P\tR" )
    print( "{:.2f}\t{:.2f}".format(P, R) )
    print( "A" )
    print( "{:.2f}".format(float(float(a_count)/all_count)) )

def test_maxent(classifier, features_set, data):
    print( "testing..." )

    sig_out = []
    t_count = 0
    m_count = 0
    c_count = 0
    a_count = 0
    all_count = 0

    all_features_results = encode_features_with_feature_set(data, features_set)
    system_results = classifier.classify_many([x[0] for x in all_features_results])

    ind = 0
    for f, model_result in all_features_results:
        system_result = system_results.pop(0)
        if model_result == '1':
            m_count += 1
        if system_result == '1':
            t_count += 1
            if model_result == '1':
                c_count += 1
        if print_error and model_result != system_result:
            t_sentence = data[ind][4][:]
            t_sentence[data[ind][3]] = '[' + t_sentence[data[ind][3]] + ']'
            print( ' '.join(t_sentence) )
            print( 'model: ' + model_result )
            print( 'system: ' + system_result )
            print( [x for x in data[ind][0] if x in features_set] )
            print
        all_count += 1
        if system_result == model_result:
            a_count += 1
            sig_out.append('{}\t{}\t{}'.format(system_result, model_result, '1'))
        else:
            sig_out.append('{}\t{}\t{}'.format(system_result, model_result, '0'))
        ind += 1
    precision_recall_accuracy([t_count, m_count, c_count, a_count, all_count])
    return sig_out

def test_maxent_prob(classifier, features_set, data):
    print( "testing..." )
    sig_out = []
    all_features_results = encode_features_with_feature_set(data, features_set)
    system_results_ori = classifier.prob_classify_many([x[0] for x in all_features_results])
    for t in prob_list:
        print( "Probability > {}".format(t) )
        t_count = 0
        m_count = 0
        c_count = 0
        a_count = 0
        all_count = 0
        ind = 0
        system_results = system_results_ori[:]
        for f, model_result in all_features_results:
            system_result = '1' if system_results.pop(0).prob('1') > t else '0'
            if model_result == '1':
                m_count += 1
            if system_result == '1':
                t_count += 1
                if model_result == '1':
                    c_count += 1
            if print_error and model_result != system_result:
                t_sentence = data[ind][4][:]
                t_sentence[data[ind][3]] = '[' + t_sentence[data[ind][3]] + ']'

            all_count += 1
            if system_result == model_result:
                a_count += 1
                sig_out.append('1')
            else:
                sig_out.append('0')
            ind += 1
        precision_recall_accuracy([t_count, m_count, c_count, a_count, all_count])
    return sig_out

def find_node_from_list(children_list, node_list):
    for x in children_list:
        if x.label in node_list:
            return x

def find_head_of_NP(n):
    head = None
    n_children_reverse = n.children[:]
    n_children_reverse.reverse()
    if len(n.children)>0:
        if n.children[-1].label == 'POS':
            head = n.children[-1]
        if not head:
            head = find_node_from_list(n_children_reverse, ['NN', 'NNP', 'NNPS', 'NNS', 'NX', 'POS', 'JJR'])
        if not head:
            head = find_node_from_list(n.children, ['NP'])
        if not head:
            head = find_node_from_list(n_children_reverse, ['ADJP', 'PRN'])
        if not head:
            head = find_node_from_list(n_children_reverse, ['CD'])
        if not head:
            head = find_node_from_list(n_children_reverse, ['JJ', 'JJS', 'RB', 'QP'])
        if not head:
            head = n.children[-1]
    return head

def find_head(n):
    out_node = None
    if n.label in FIND_HEAD_DICT.keys():
        target_tuple = FIND_HEAD_DICT[n.label]
        n_children = n.children[:]
        if target_tuple[0]:
            n_children.reverse()
        for l in target_tuple[1]:
            for x in n_children:
                if x.label == l:
                    out_node = x
            if out_node:
                break
    elif n.label == 'NP':
        out_node = find_head_of_NP(n)

    if not out_node:
        if n.children:
            out_node = n.children[0]
        else:
            return None
    if out_node.label in FIND_HEAD_DICT.keys() or out_node.label == 'NP':
        return find_head(out_node)
    elif out_node.word:
        return out_node
    else:
        return None


def convert_noun_pos(pos_label, word):
    if pos_label == 'PRP':
        if word.lower() in ['i', 'you', 'we', 'they']:
            pos_label = 'NNS'
        else:
            pos_label = 'NN'
    elif pos_label == 'NNP':
        pos_label = 'NN'
    elif pos_label == 'NNPS':
        pos_label = 'NNS'
    return pos_label


def get_node_with_pos(n, lpos, tagger_output, get_word=False):
    temp = None
    if n.label in ['VP', 'NP', 'ADVP', 'ADJP']:
        temp = find_head(n)
    if temp:
        if tagger_output:
            if convert_noun:
                return n.label + '|' + convert_noun_pos(lpos[temp.word_index-1][1], lpos[temp.word_index-1][0])
            else:
                if get_word and n.label == 'NP':
                # if get_word:
                    return n.label + '|' + lpos[temp.word_index-1][0]
                else:
                    return n.label + '|' + lpos[temp.word_index-1][1]
        else:
            if convert_noun:
                return n.label + '|' + convert_noun_pos(temp.label, lpos[temp.word_index-1][0])
            else:
                return n.label + '|' + temp.label
    else:
        return n.label


def find_s(n, path):
    path.append(n.id)
    if n.parent_n:
        if n.parent_n.label in ['S', 'SBAR', 'SINV', 'FRAG']:
            return (n.parent_n, path)
        else:
            return find_s(n.parent_n, path)
    else:
        return (n, path)

def get_trigram(i, l, prefix):
    prev_t = l[i-1] if i-1>=0 else '<start>'
    next_t = l[i+1] if i+1<len(l) else '<end>'
    mid_t = l[i] if i<len(l) else '<mid>'
    return (prefix + ':' + '_'.join([prev_t, mid_t, next_t]), [prev_t, mid_t, next_t])

def process_files_whole_sentence(sentences, sentences_lpos, sentences_trees):
    all_instances = []
    sentence_id = 0
    tagger_output = True
    for sentence, lpos, tree in zip(sentences, sentences_lpos, sentences_trees):
        sentence_ori = sentence.replace('*', '').split(' ')
        sentence = sentence.split(' ')
        lpos_all = [x.split('|') for x in lpos.split(' ')]
        lpos = [x.split('|')[1] for x in lpos.split(' ')]
        nodes = stanford_parse.parse_tree(tree)

        instance_features = []

        for i in range(0, len(sentence)):
            if include_wordtrigram:
                wordtrigram = get_trigram(i, sentence_ori, 'WORDTRI')
                instance_features.append(wordtrigram[0])
                instance_features.append('WORDTRI:' + '_'.join(wordtrigram[1][:2]))
                instance_features.append('WORDTRI:' + '_'.join(wordtrigram[1][1:]))
                instance_features.append('WORDTRI:' + wordtrigram[1][1])

            if include_postrigram:
                postrigram = get_trigram(i, lpos, 'POSTRI')
                instance_features.append(postrigram[0])
                instance_features.append('POSTRI:' + '_'.join(postrigram[1][:2]))
                instance_features.append('POSTRI:' + '_'.join(postrigram[1][1:]))
                instance_features.append('POSTRI:' + postrigram[1][1])

        if include_nodetrigram:
            for n in nodes:
                if n.label in ['S', 'SBAR', 'SINV', 'FRAG']:
                    s_node = n
                    node_indexs = range(len(s_node.children))
                    for i in node_indexs:
                        temp_trigram = []
                        temp_word_trigram = []
                        temp_node_trigram = []
                        if i-1>=0:
                            temp_trigram.append(get_node_with_pos(s_node.children[i-1], lpos_all, tagger_output))
                            temp_word_trigram.append(get_node_with_pos(s_node.children[i-1], lpos_all, tagger_output, True))
                            temp_node_trigram.append(s_node.children[i-1])
                        else:
                            temp_trigram.append('<start>')
                            temp_word_trigram.append('<start>')
                        temp_trigram.append(get_node_with_pos(s_node.children[i], lpos_all, tagger_output))
                        temp_word_trigram.append(get_node_with_pos(s_node.children[i], lpos_all, tagger_output, True))
                        temp_node_trigram.append(s_node.children[i])
                        if i+1<len(s_node.children):
                            temp_trigram.append(get_node_with_pos(s_node.children[i+1], lpos_all, tagger_output))
                            temp_word_trigram.append(get_node_with_pos(s_node.children[i+1], lpos_all, tagger_output, True))
                            temp_node_trigram.append(s_node.children[i+1])
                        else:
                            temp_trigram.append('<end>')
                            temp_word_trigram.append('<end>')

                        instance_features.append(s_node.label + '||' + '_'.join(temp_trigram[:3]))
                        instance_features.append(s_node.label + '||' + '_'.join(temp_word_trigram[:3]))

                        instance_features.append(s_node.label + '||' + '_'.join(temp_trigram[:2]))
                        instance_features.append(s_node.label + '||' + '_'.join(temp_word_trigram[:2]))

                        instance_features.append(s_node.label + '||' + '_'.join(temp_trigram[1:]))
                        instance_features.append(s_node.label + '||' + '_'.join(temp_word_trigram[1:]))

                        instance_features.append(s_node.label + '||' + '_'.join(temp_trigram[:1]))
                        instance_features.append(s_node.label + '||' + '_'.join(temp_word_trigram[:1]))

        sva_flag = '0'
        if '*' in ' '.join(sentence):
            sva_flag = '1'
        instance_features = list(set(instance_features))
        all_instances.append((instance_features, sva_flag, sentence_id, 0, sentence))
        sentence_id += 1
    return all_instances

def process_files(file_list, feature_function, filter_func=None, neg_pos_ratio=1, no_of_sentence=10000, start_from=0):
    print( "processing file {}...".format(file_list[0]) )
    sentences = open(file_list[0]).read().splitlines()
    sentences_lpos = open(file_list[1].replace('.txt', '_lpos.txt')).read().splitlines()
    sentences_trees = stanford_parse.get_trees_from_raw(open(file_list[1].replace('.txt', '_trees.txt')).read())
    if filter_func:
        sentences, sentences_lpos, sentences_trees = filter_func(sentences, sentences_lpos, sentences_trees, neg_pos_ratio, no_of_sentence, start_from)
    return feature_function(sentences, sentences_lpos, sentences_trees)

def process_training_files(file_list, feature_function, filter_func=None, neg_pos_ratio=1, no_of_sentence=10000, start_from=0, align=False):
    print( "processing file {}...".format(file_list[0]) )
    no_of_sentence_for_preprocess = 18000
    sentences = open(file_list[0]).read().splitlines()
    sentences_lpos = open(file_list[1].replace('.txt', '_lpos.txt')).read().splitlines()
    sentences_trees = stanford_parse.get_trees_from_raw(open(file_list[1].replace('.txt', '_trees.txt')).read())

    no_of_pos = no_of_sentence/(neg_pos_ratio+1)
    no_of_neg = no_of_pos * neg_pos_ratio

    if filter_func:
        sentences, sentences_lpos, sentences_trees = filter_func(sentences, sentences_lpos, sentences_trees, 1, no_of_sentence_for_preprocess, start_from)
    all_instances = feature_function(sentences, sentences_lpos, sentences_trees)
    if align:
        aligned_file = open(file_list[1].replace('.txt', '_align.txt')).read().splitlines()
        aligned_file = [x.split(' ') for x in aligned_file]

    all_instances = dict([((x[2], x[3]), x) for x in all_instances])

    filtered_all_instances = []
    neg_instances = []
    for x in all_instances.keys():
        if all_instances[x][1] == '1':
            if x[0]%2:                      # odd number, incorrect
                aligned_sent = x[0]-1
            else:
                aligned_sent = x[0]+1
            aligned_ind = int(aligned_file[x[0]][x[1]])
            if (aligned_sent, aligned_ind) in all_instances.keys():
                current_instance = all_instances[x]
                aligned_instance = all_instances[(aligned_sent, aligned_ind)]
                if current_instance[1] != aligned_instance[1]:
                    current_local_features = [x for x in current_instance[0] if "WORDTRI:" in x or "POSTRI:" in x]
                    current_parse_features = [x for x in current_instance[0] if "WORDTRI:" not in x and "POSTRI:" not in x]
                    aligned_local_features = [x for x in aligned_instance[0] if "WORDTRI:" in x or "POSTRI:" in x]
                    aligned_parse_features = [x for x in aligned_instance[0] if "WORDTRI:" not in x and "POSTRI:" not in x]
                    temp = list(set(current_parse_features) - set(aligned_parse_features))
                    aligned_instance[0] = list(set(aligned_parse_features) - set(current_parse_features)) + aligned_local_features
                    current_instance[0] = temp + current_local_features
                    if len(filtered_all_instances) < no_of_pos:
                        filtered_all_instances.append(current_instance)
                    if len(neg_instances) < no_of_neg:
                        neg_instances.append(aligned_instance)
                    else:
                        break

    print( "Training data (filtered):" )
    print( "no of pos: {}".format(len(filtered_all_instances)) )
    print( "no of neg: {}".format(len(neg_instances)) )
    filtered_all_instances += neg_instances
    return filtered_all_instances


def training_data_filter(sentences, sentences_lpos, sentences_trees, neg_pos_ratio, no_of_sentence, start_from):
    new_sentences = []
    new_sentences_lpos = []
    new_sentences_trees = []
    no_of_pos = round(no_of_sentence/(neg_pos_ratio+1))
    no_of_neg = round(no_of_pos * neg_pos_ratio)
    print( "Training data (preprocess):" )
    count = 0
    pos_count = 0
    neg_count = 0
    for sentence, lpos, tree in zip(sentences[start_from:], sentences_lpos[start_from:], sentences_trees[start_from:]):
        if no_of_pos == 0 and no_of_neg == 0:
            break
        if '*' in sentence:
            if no_of_pos > 0:
                new_sentences.append(sentence)
                new_sentences_lpos.append(lpos)
                new_sentences_trees.append(tree)
                no_of_pos -= 1
                pos_count += 1
        else:
            if no_of_neg > 0:
                new_sentences.append(sentence)
                new_sentences_lpos.append(lpos)
                new_sentences_trees.append(tree)
                no_of_neg -= 1
                neg_count += 1
        count += 1

    print( "no of pos: {}".format(pos_count) )
    print( "no of neg: {}".format(neg_count)         )
    print( len(new_sentences) )
    print( start_from )
    print( count     )
    print( start_from + count )
    return (new_sentences, new_sentences_lpos, new_sentences_trees)

def nucle_data_filter(sentences, sentences_lpos, sentences_trees, neg_pos_ratio, no_of_sentence, start_from):
    return (sentences, sentences_lpos, sentences_trees)

def show_most_informative_features_binary(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    print( clf.classes_ )
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print( "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2) )

if __name__ == '__main__':

    include_wordtrigram = False
    include_postrigram = False
    include_nodetrigram = True

    feauture_function = process_files_whole_sentence

    # pos_neg_ratio
    r = 1

    print( "ratio: {}".format(r) )
    print( 'word: {}'.format(include_wordtrigram) )
    print( 'pos: {}'.format(include_postrigram) )
    print( 'node: {}'.format(include_nodetrigram) )
    all_instances = process_files(DATA['nucle_frag_f'], feauture_function, training_data_filter, neg_pos_ratio=r)
    features_set, classifier = train_maxent(all_instances)


    show_most_informative_features_binary(classifier._vectorizer, classifier._clf, n=100)

    all_instances = process_files(DATA['nucle_frag_f'], feauture_function)

    sig_out = test_maxent(classifier, features_set, all_instances)
