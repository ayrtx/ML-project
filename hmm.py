import os
import math
import copy

#enter which version (SG, EN, CN, FR)
ver = 'FR'
script_dir = os.path.dirname('__file__')
train_file_path = os.path.join(script_dir, ver + "/train")
test_file_path = os.path.join(script_dir, ver + "/dev.in")

tag_all = ['B-positive', 'I-positive', 'B-negative', 'I-negative', 'B-neutral', 'I-neutral', 'O']

#emission parameters
def e(file, k):
    tag_count = {'B-positive':0, 'I-positive':0, 'B-negative':0, 'I-negative':0, 'B-neutral':0, 'I-neutral':0, 'O':0}
    lines = []
    for line in file:
        if line != '\n':
            splited = line.split(' ')
            lines.append(((' '.join(splited[:-1])), splited[-1][:-1]))
    for i in lines:
        tag_count [i[-1]] += 1
    words_tag_count = {}
    for i in lines:
        if i not in words_tag_count.keys():
            words_tag_count[i] = 1
        else:
            words_tag_count[i] += 1
    words_mle = copy.deepcopy(words_tag_count)

    for i in words_mle:
        words_mle[i] = words_mle[i]/tag_count[i[-1]]

    words_tag_count_fix = copy.deepcopy(words_tag_count)
    for i in tag_count:
        words_tag_count_fix[('#UNK#', i)] = 0
    word_count = {}
    for key, value in words_tag_count.items():
        if key[0] in word_count:
            word_count[key[0]] += value
        else:
            word_count[key[0]] = value
    words_to_unk = []
    for key, value in word_count.items():
        if value < k:
            words_to_unk.append(key)
    for i in words_tag_count:
        if i[0] in words_to_unk:
            words_tag_count_fix[('#UNK#', i[-1])] += words_tag_count[i]
            del words_tag_count_fix[i]
    words_mle_fix = copy.deepcopy(words_tag_count_fix)
    for i in words_mle_fix:
        words_mle_fix[i] = words_mle_fix[i]/tag_count[i[-1]]
    return words_mle_fix


############
#  PART 2  #
############

#simple analysis system
def arg_max(input_file, data):
    dict = {}
    for i in data:
        if i[0] not in dict:
            dict[i[0]] = ""
    
    for word in dict:
        max_prob = 0
        for pair in data:
            if pair[0] == word:
                if data[pair] > max_prob:
                    max_prob = data[pair]
                    dict[word] = pair[-1]
                    
    return output_p2(input_file, dict)

#writing to p2 output                    
def output_p2(input_file, data):
    path = ver + '/dev.p2.out'
    output_file = open(path, 'w').close()
    output_file = open(path, 'w', encoding="utf8")
    
    for line in input_file:
        if line != "\n":
            line = line.strip()
            try:
                prediction = data[line]
    
            except:
                prediction = data['#UNK#']
                
            output_file.write(line + ' ' + prediction + '\n')
            
        elif line == "\n":
            output_file.write('\n')
    
    output_file.close()
    return "p2 done"

#running simple analysis system
test_file = open(test_file_path, encoding="utf8")
train_file = open(train_file_path, encoding="utf8")
arg_max(test_file, e(train_file, 3))
train_file.close()
test_file.close()

############
#  PART 3  #
############

#transition parameters
def q(file):
    count = {}
    dict = {}
    state = "start"
    old_label = ""
    current_label = "START"
    for line in file:
        if state == "start":
            if line != "\n":
                try:
                    count["START"] += 1
                except:
                    count["START"] = 1
                    
                current_label = line.strip().split()[-1]
                
                try:
                    dict[("START", current_label)] += 1
                except:
                    dict[("START", current_label)] = 1
                    
                try:
                    count[current_label] += 1
                except:
                    count[current_label] = 1
                    
                state = "neither"
            
        elif state == "neither":
            if line == "\n":
                old_label = current_label
                    
                try:
                    dict[(old_label, "STOP")] += 1
                except:
                    dict[(old_label, "STOP")] = 1
                state = "start"
            
            elif line != "\n":
                old_label = current_label
                
                current_label = line.strip().split()[-1]
                
                try:
                    count[current_label] += 1
                except:
                    count[current_label] = 1
                    
                try:
                    dict[(old_label, current_label)] += 1
                except:
                    dict[(old_label, current_label)] = 1
                
                state = "neither"
  
    for i in dict:
        dict[i] = float(dict[i])/float(count[i[0]])
    
    label_possible = []
    for key in dict:
        if key[0] == "START":
            label_possible.append(key[1])
    for i in tag_all:
        if i not in label_possible:
            dict[('START', i)] = 0

    return dict

#viberti
def viberti(file, transition, emission):     
    path = ver + '/dev.p3.out'
    output_file = open(path, 'w').close()
    output_file = open(path, 'w', encoding="utf8")
    
    unknown = {key:value for key, value in emission.items() if key[0] == "#UNK#"}
    unknown_emission = {}
    for pair in unknown:
        unknown_emission[pair[1]] = unknown[pair]
        
    emission = {key:value for key, value in emission.items() if key[0] != "#UNK#"}

    all_sentences = []
    sentence = []
    
    for line in file:
        if line != "\n":
            sentence.append(line.strip())
        elif line == "\n":
            all_sentences.append(sentence)
            sentence = []
    
    for sentence in all_sentences:

        backtrace = []

        old_prob = {}
        current_prob = {}
        
        for level in range(len(sentence)):
            
            word = sentence[level]
            word_exists = False
            
            emission_prob = {}
            old_prob = current_prob
            current_prob = {}
            
            for pair in emission:
                if pair[0] == word:
                    word_exists = True
                    emission_prob[pair[1]] = emission[pair]
            
            if word_exists != True:
                emission_prob = unknown_emission
            
            if level == 0:
                for label in emission_prob:
                    try:
                        current_prob[label] = emission_prob[label] * transition[("START", label)]
                        try:
                            backtrace[level][label] = "START"
                        except:
                            backtrace.append({label: "START"})
                    except:
                        pass
            
            else:
                for label in emission_prob:
                    max_prob = "DEFAULT"
#                     old_prob = {key:value for key, value in old_prob.items() if value != 0 and value != "NA"}
                    for old_label in old_prob:
                        try:
                            prob = old_prob[old_label] * emission_prob[label] * transition[(old_label, label)]
                            if max_prob == "DEFAULT":
                                max_prob = prob
                                current_prob[label] = prob
                                try:
                                    backtrace[level][label] = old_label
                                except:
                                    backtrace.append({label: old_label})
                            
                            elif prob > max_prob:
                                max_prob = prob
                                current_prob[label] = prob
                                try:
                                    backtrace[level][label] = old_label
                                except:
                                    backtrace.append({label: old_label})
                            
                        except KeyError as e:
                            pass
                if len(current_prob) == 0:
                    current_prob = {'O':1.0}
                    backtrace.append({'O': old_label})
                        

        stop_label = "DEFAULT"
        for label in backtrace[-1]:
            max_prob = "DEFAULT"
            try:
                prob = current_prob[label] * transition[(label, "STOP")]
                if max_prob == "DEFAULT":
                    max_prob = prob
                    stop_label = label

                elif prob > max_prob:
                    max_prob = prob
                    stop_label = label
                    
            except KeyError as e:
                pass
        
        backtrace.append({"STOP": stop_label})
            
        backtrace.reverse()
        
        if backtrace[0]["STOP"] == "DEFAULT":
            print(sentence)
            print(backtrace)
            
        the_label = "STOP"
        prediction = ["STOP"]
        for labels in backtrace:
            the_label = labels[the_label]
            prediction.append(the_label)
        
        prediction.reverse()
        
        output_p3(output_file, sentence, prediction)
    
    output_file.close()
    return "Testing done"

#writing to p3 output
def output_p3(output_file, sentence, prediction):
    for n in range(len(sentence)):
        output_file.write(sentence[n] + ' ' + prediction[n+1] + '\n')
    output_file.write('\n')

train_file = open(train_file_path, encoding="utf8")
emission_dict = e(train_file, 3)
train_file.close()

train_file = open(train_file_path, encoding="utf8")
transition_dict = q(train_file)
train_file.close()

#running viberti
test_file = open(test_file_path, encoding="utf8")
viberti(test_file, transition_dict, emission_dict)
test_file.close()

############
#  PART 4  #
############

#max-marginal
def alt_max_marginal(file, transition, emission):
    all_sentences = []
    sentence = []
    predicted = []
    
    for line in file:
        if line != "\n":
            sentence.append(line.strip())
        elif line == "\n":
            all_sentences.append(sentence)
            sentence = []
            
    transition_start = {}
    transition_stop = {}
    transition_others = {}
    for i in transition:
        if 'START' in i:
            transition_start[i] = transition[i]
        elif 'STOP' in i:
            transition_stop[i] = transition[i]
        else:
            transition_others[i] = transition[i]
            
    tags = ['B-positive', 'I-positive', 'B-negative', 'I-negative', 'B-neutral', 'I-neutral', 'O']
    
    for sentence in all_sentences:
        forward = []
        forward_each = []
        
        #forward base case
        for key, value in transition_start.items():
            forward_each.append([key[1], value])
        forward.append(forward_each)
        
        #forward recursive case
        for i in range(1, len(sentence)):
            forward_each = []
            for tag in tags:
                forward_sum = 0
                for j in forward[-1]:
                    try:
                        if (sentence[i-1], j[0]) in emission:
                            forward_sum += j[1] * transition_others[(j[0], tag)] * emission[(sentence[i-1], j[0])]
                        else:
                            exist = False
                            for k in tags:
                                if(sentence[i-1], k) in emission:
                                    exist = True
                                    break
                            if not exist:
                                forward_sum += j[1] * transition_others[(j[0], tag)] * emission[('#UNK#', j[0])]
                    except Exception as e:
                        pass
            
                if forward_sum != 0:
                    forward_each.append([tag, forward_sum])
            forward.append(forward_each)
            
        backward = []
        backward_each = []
        
        #backward base case
        for key, value in transition_stop.items():
            if (sentence[-1], key[0]) in emission:
                backward_each.append([key[0], value * emission[(sentence[-1], key[0])]])
            else:
                exist = False
                for k in tags:
                    if(sentence[-1], k) in emission:
                        exist = True
                        break
                if not exist:
                    backward_each.append([key[0], value * emission[('#UNK#', key[0])]])         
        backward.append(backward_each)
                    
        #backward recursive case
        for i in range(2, len(sentence) + 1):
            backward_each = []
            for tag in tags:
                backward_sum = 0
                for j in backward[0]:
                    if (tag, j[0]) in transition_others:
                        try:
                            backward_sum += transition_others[(tag, j[0])] * emission[(sentence[-i], tag)] * j[1]
                        except Exception as e:
                            exist = False
                            for k in tags:
                                if(sentence[-i], k) in emission:
                                    exist = True
                                    break
                            if not exist:
                                backward_sum += transition_others[(tag, j[0])] * emission[('#UNK#', tag)] * j[1]
                if backward_sum != 0:
                    backward_each.append([tag, backward_sum])
            backward.insert(0, backward_each)
            
        #predicting labels
        predicted_each = []
        for i in range(0, len(forward)):
            max_val = ['tag', 0]
            for j in forward[i]:
                for k in backward[i]:
                    if j[0] == k[0]:
                        if j[1] * k[1] > max_val[1]:
                            max_val = [j[0], j[1] * k[1]]
                        break
            predicted_each.append(max_val)
            
        #to get sum of (alpha x beta) through all labels for each word
#         for i in range(0, len(forward)):
#             summ = 0
#             for j in forward[i]:
#                 for k in backward[i]:
#                     if j[0] == k[0]:
#                         summ += j[1] * k[1]
#                         break
#             print (summ)
            
        predicted.append(predicted_each)
    
    path = ver + '/dev.p4.out'
    output_file = open(path, 'w').close()
    output_file = open(path, 'w', encoding="utf8")
    for i in range(0, len(all_sentences)):
        for j in range(0, len(all_sentences[i])):
            output_file.write(all_sentences[i][j] + ' ' + predicted[i][j][0] + '\n')
        output_file.write('\n')
    output_file.close()

#running max-marginal
test_file = open(test_file_path, encoding="utf8")
alt_max_marginal(test_file, transition_dict, emission_dict)
test_file.close()