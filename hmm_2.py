import os
import math
import copy

#p5s.out, p5e.out, p5n.out are intermediate files and can be removed once the code has finished running

#enter which version (SG, EN)
ver = 'EN'
#enter True if using test.in or False if not
test_p5 = False
script_dir = os.path.dirname('__file__')
train_file_path = os.path.join(script_dir, ver + "/train")
if test_p5:
    test_file_path = os.path.join(script_dir, ver + "/test.in")
else:
    test_file_path = os.path.join(script_dir, ver + "/dev.in")


tag_all = ['B-positive', 'I-positive', 'B-negative', 'I-negative', 'B-neutral', 'I-neutral', 'O']
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

#posterior-viberti
def posterior_viberti(file, transition, emission):
    if test_p5:
        path = ver + '/test.p5n.out'
    else:
        path = ver + '/dev.p5n.out'
    output_file = open(path, 'w').close()
    output_file = open(path, 'w', encoding="utf8")
    
    unknown = {key:value for key, value in emission.items() if key[0] == "#UNK#"}
    unknown_emission = {}
    for pair in unknown:
        unknown_emission[pair[1]] = unknown[pair]
        
    emission_new = {key:value for key, value in emission.items() if key[0] != "#UNK#"}

    all_sentences = []
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
            
    for line in file:
        if line != "\n":
            sentence.append(line.strip())
        elif line == "\n":
            all_sentences.append(sentence)
            sentence = []
    
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

        pom = 0
        for alpha in forward[-1]:
            for key, value in transition_stop.items():
                if alpha[0] == key[0]:
                    pom += alpha[1] * value
                    
        backtrace = []

        old_prob = {}
        current_prob = {}
        
        for level in range(len(sentence)):
            
            word = sentence[level]
            word_exists = False
            
            emission_prob = {}
            old_prob = current_prob
            current_prob = {}
            
            for pair in emission_new:
                if pair[0] == word:
                    word_exists = True
                    emission_prob[pair[1]] = emission_new[pair]
            
            if word_exists != True:
                emission_prob = unknown_emission
            
            if level == 0:
                for label in emission_prob:
                    try:
                        if transition[("START", label)] > 0:
                            current_prob[label] = 1
                        else:
                            current_prob[label] = 0
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
                            if transition[(old_label, label)] > 0:
                                for alpha in forward[level]:
                                    if alpha[0] == label:
                                        for beta in backward[level]:
                                            if beta[0] == label:
                                                prob = old_prob[old_label] * (alpha[1]*beta[1]/pom) * 1
                                                break
                                        break 
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
                if transition[(label, "STOP")] > 0:
                    prob = current_prob[label] * 1
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
        output_p5(output_file, sentence, prediction)
    
    output_file.close()
    return "Testing done"

def e(file, k):
    tag_count = {'B-positive':0, 'I-positive':0, 'B-negative':0, 'I-negative':0, 'B-neutral':0, 'I-neutral':0, 'O':0}
    unk_count = {'B-positive':0, 'I-positive':0, 'B-negative':0, 'I-negative':0, 'B-neutral':0, 'I-neutral':0, 'O':0}
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
    # print (words_mle)

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
            unk_count[i[-1]] += words_tag_count[i]
            words_tag_count_fix[('#UNK#', i[-1])] += words_tag_count[i]
            
    words_mle_fix = copy.deepcopy(words_tag_count_fix)
    for i in words_mle_fix:
        if i[-1] == '#UNK#':
            words_mle_fix[i] = words_mle_fix[i]/unk_count[i[-1]]
        else:
            words_mle_fix[i] = words_mle_fix[i]/tag_count[i[-1]]
    return words_mle_fix

#writing to p5 output
def output_p5(output_file, sentence, prediction):
    for n in range(len(sentence)):
        output_file.write(sentence[n] + ' ' + prediction[n+1] + '\n')
    output_file.write('\n')

train_file = open(train_file_path, encoding="utf8")
emission_dict = e(train_file, 3)
train_file.close()

train_file = open(train_file_path, encoding="utf8")
transition_dict = q(train_file)
train_file.close()

#running posterior-viberti
if test_p5:
    test_file_path = os.path.join(script_dir, ver + "/test.in")
else:
    test_file_path = os.path.join(script_dir, ver + "/dev.in")

test_file = open(test_file_path, encoding="utf8")
posterior_viberti(test_file, transition_dict, emission_dict)
test_file.close()



tag_all = ['B', 'I', 'O']
#posterior-viberti for entity
def posterior_viberti_e(file, transition, emission):
    if test_p5:
        path = ver + '/test.p5e.out'
    else:
        path = ver + '/dev.p5e.out'
    output_file = open(path, 'w').close()
    output_file = open(path, 'w', encoding="utf8")
    
    unknown = {key:value for key, value in emission.items() if key[0] == "#UNK#"}
    unknown_emission = {}
    for pair in unknown:
        unknown_emission[pair[1]] = unknown[pair]
        
    emission_new = {key:value for key, value in emission.items() if key[0] != "#UNK#"}

    all_sentences = []
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
            
    tags = ['B', 'I', 'O']
            
    for line in file:
        if line != "\n":
            sentence.append(line.strip())
        elif line == "\n":
            all_sentences.append(sentence)
            sentence = []
    
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

        pom = 0
        for alpha in forward[-1]:
            for key, value in transition_stop.items():
                if alpha[0] == key[0]:
                    pom += alpha[1] * value
                    
        backtrace = []

        old_prob = {}
        current_prob = {}
        
        for level in range(len(sentence)):
            
            word = sentence[level]
            word_exists = False
            
            emission_prob = {}
            old_prob = current_prob
            current_prob = {}
            
            for pair in emission_new:
                if pair[0] == word:
                    word_exists = True
                    emission_prob[pair[1]] = emission_new[pair]
            
            if word_exists != True:
                emission_prob = unknown_emission
            
            if level == 0:
                for label in emission_prob:
                    try:
                        if transition[("START", label)] > 0:
                            current_prob[label] = 1
                        else:
                            current_prob[label] = 0
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
                            if transition[(old_label, label)] > 0:
                                for alpha in forward[level]:
                                    if alpha[0] == label:
                                        for beta in backward[level]:
                                            if beta[0] == label:
                                                prob = old_prob[old_label] * (alpha[1]*beta[1]/pom) * 1
                                                break
                                        break 
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
                if transition[(label, "STOP")] > 0:
                    prob = current_prob[label] * 1
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
        output_p5(output_file, sentence, prediction)
    
    output_file.close()
    return "Testing done"

def e_e(file, k):
    tag_count = {'B':0, 'I':0, 'O':0}
    unk_count = {'B':0, 'I':0, 'O':0}
    lines = []
    for line in file:
        if line != '\n':
            splited = line.split(' ')
            lines.append(((' '.join(splited[:-1])), splited[-1][:-1].split('-')[0]))
            
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
    # print (words_mle)

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
            unk_count[i[-1]] += words_tag_count[i]
            words_tag_count_fix[('#UNK#', i[-1])] += words_tag_count[i]
            
    words_mle_fix = copy.deepcopy(words_tag_count_fix)
    for i in words_mle_fix:
        if i[-1] == '#UNK#':
            words_mle_fix[i] = words_mle_fix[i]/unk_count[i[-1]]
        else:
            words_mle_fix[i] = words_mle_fix[i]/tag_count[i[-1]]
    return words_mle_fix

def q_e(file):
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
                    
                current_label = line.strip().split()[-1].split('-')[0]
                
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
                
                current_label = line.strip().split()[-1].split('-')[0]
                
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

#writing to p5 output
def output_p5(output_file, sentence, prediction):
    for n in range(len(sentence)):
        output_file.write(sentence[n] + ' ' + prediction[n+1] + '\n')
    output_file.write('\n')

train_file = open(train_file_path, encoding="utf8")
emission_dict = e_e(train_file, 3)
train_file.close()

train_file = open(train_file_path, encoding="utf8")
transition_dict = q_e(train_file)
train_file.close()

#running posterior-viberti for entity
test_file = open(test_file_path, encoding="utf8")
posterior_viberti_e(test_file, transition_dict, emission_dict)
test_file.close()

tag_all = ['positive', 'negative', 'neutral', 'O']
#posterior-viberti for sentiment
def posterior_viberti_s(file, transition, emission):
    if test_p5:
        path = ver + '/test.p5s.out'
    else:
        path = ver + '/dev.p5s.out'
    output_file = open(path, 'w').close()
    output_file = open(path, 'w', encoding="utf8")
    
    unknown = {key:value for key, value in emission.items() if key[0] == "#UNK#"}
    unknown_emission = {}
    for pair in unknown:
        unknown_emission[pair[1]] = unknown[pair]
        
    emission_new = {key:value for key, value in emission.items() if key[0] != "#UNK#"}

    all_sentences = []
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
            
    tags = ['positive', 'negative', 'neutral', 'O']
            
    for line in file:
        if line != "\n":
            sentence.append(line.strip())
        elif line == "\n":
            all_sentences.append(sentence)
            sentence = []
    
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

        pom = 0
        for alpha in forward[-1]:
            for key, value in transition_stop.items():
                if alpha[0] == key[0]:
                    pom += alpha[1] * value
                    
        backtrace = []

        old_prob = {}
        current_prob = {}
        
        for level in range(len(sentence)):
            
            word = sentence[level]
            word_exists = False
            
            emission_prob = {}
            old_prob = current_prob
            current_prob = {}
            
            for pair in emission_new:
                if pair[0] == word:
                    word_exists = True
                    emission_prob[pair[1]] = emission_new[pair]
            
            if word_exists != True:
                emission_prob = unknown_emission
            
            if level == 0:
                for label in emission_prob:
                    try:
                        if transition[("START", label)] > 0:
                            current_prob[label] = 1
                        else:
                            current_prob[label] = 0
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
                            if transition[(old_label, label)] > 0:
                                for alpha in forward[level]:
                                    if alpha[0] == label:
                                        for beta in backward[level]:
                                            if beta[0] == label:
                                                prob = old_prob[old_label] * (alpha[1]*beta[1]/pom) * 1
                                                break
                                        break 
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
                if transition[(label, "STOP")] > 0:
                    prob = current_prob[label] * 1
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
        output_p5(output_file, sentence, prediction)
    
    output_file.close()
    return "Testing done"

def e_s(file, k):
    tag_count = {'positive':0, 'negative':0, 'neutral':0, 'O':0}
    unk_count = {'positive':0, 'negative':0, 'neutral':0, 'O':0}
    lines = []
    for line in file:
        if line != '\n':
            splited = line.split(' ')
            lines.append(((' '.join(splited[:-1])), splited[-1][:-1].split('-')[-1]))
            
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
    # print (words_mle)

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
            unk_count[i[-1]] += words_tag_count[i]
            words_tag_count_fix[('#UNK#', i[-1])] += words_tag_count[i]
            
    words_mle_fix = copy.deepcopy(words_tag_count_fix)
    for i in words_mle_fix:
        if i[-1] == '#UNK#':
            words_mle_fix[i] = words_mle_fix[i]/unk_count[i[-1]]
        else:
            words_mle_fix[i] = words_mle_fix[i]/tag_count[i[-1]]
    return words_mle_fix

def q_s(file):
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
                    
                current_label = line.strip().split()[-1].split('-')[-1]
                
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
                
                current_label = line.strip().split()[-1].split('-')[-1]
                
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

#writing to p5 output
def output_p5(output_file, sentence, prediction):
    for n in range(len(sentence)):
        output_file.write(sentence[n] + ' ' + prediction[n+1] + '\n')
    output_file.write('\n')

train_file = open(train_file_path, encoding="utf8")
emission_dict = e_s(train_file, 3)
train_file.close()

train_file = open(train_file_path, encoding="utf8")
transition_dict = q_s(train_file)
train_file.close()

#running posterior-viberti for sentiment
test_file = open(test_file_path, encoding="utf8")
posterior_viberti_s(test_file, transition_dict, emission_dict)
test_file.close()

if test_p5:
    entity_file_path = os.path.join(script_dir, ver + "/test.p5e.out")
    sentiment_file_path = os.path.join(script_dir, ver + "/test.p5s.out")
    normal_file_path = os.path.join(script_dir, ver + "/test.p5n.out")
else:
    entity_file_path = os.path.join(script_dir, ver + "/dev.p5e.out")
    sentiment_file_path = os.path.join(script_dir, ver + "/dev.p5s.out")
    normal_file_path = os.path.join(script_dir, ver + "/dev.p5n.out")

entity_file_list = []
entity_file = open(entity_file_path, encoding="utf8")
for line in entity_file:
    splited = line.split(' ')
    entity_file_list.append(((' '.join(splited[:-1])), splited[-1][:-1]))
entity_file.close()

sentiment_file_list = []
sentiment_file = open(sentiment_file_path, encoding="utf8")
for line in sentiment_file:
    splited = line.split(' ')
    sentiment_file_list.append(((' '.join(splited[:-1])), splited[-1][:-1]))
sentiment_file.close()

normal_file_list = []
normal_file = open(normal_file_path, encoding="utf8")
for line in normal_file:
    splited = line.split(' ')
    normal_file_list.append(((' '.join(splited[:-1])), splited[-1][:-1]))
normal_file.close()

if test_p5:
    final_file_path = os.path.join(script_dir, ver + "/test.p5.out")
else:
    final_file_path = os.path.join(script_dir, ver + "/dev.p5.out")

final_file = open(final_file_path, 'w', encoding="utf8")
for i in range(0, len(normal_file_list)):
    if entity_file_list[i][1] == 'O' or sentiment_file_list[i][1] == 'O':
        if entity_file_list[i][1] == sentiment_file_list[i][1]:
            final_file.write(normal_file_list[i][0] + ' O\n')
        else:
            final_file.write(normal_file_list[i][0] + ' ' + normal_file_list[i][1] + '\n')
    elif entity_file_list[i][0] == '':
        final_file.write('\n')
    else:
        final_file.write(normal_file_list[i][0] + ' ' + entity_file_list[i][1] + '-' + sentiment_file_list[i][1] + '\n')
final_file.close()
