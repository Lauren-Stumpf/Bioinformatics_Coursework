"""
To run call the function EM(alphabet, n, S) where alphabet is a string of the unique symbols in the alphabet (eg. 'abcdefg'), n is the number of hidden states (eg. 4), 
S is a list of sequences eg. ['abc', 'efg', 'hij']. If one sequence it still needs to be passed in as a list eg. ['abc']

Note, the alphabet shouldn't contain duplicate letters eg. a and aa as two separate symbols
Note, the order the alphabet is passed in will be the order of the emission matrix b 
The function returns a (M) - size (n,n), b (E) - size(n, len(alphabet)), pi (P) - size(n,)

"""



import numpy as np 
import random

#Function to map the alphabet to numbers, which is the input our implementation takes
def map_alphabet_to_numbers(alphabet):
  alphabet_dictionary = {}
  count = 0
  for letter in alphabet:
      #if letter not in alphabet_dictionary:
      alphabet_dictionary[letter] = count
      count += 1

  return alphabet_dictionary


#Function to convert a sequence of letters to a sequence of numbers using the alphabet_dictionary provided
def convert_letter_sequence_to_number_sequence(alphabet_dictionary, letter_sequence_list):
  final_sequence = []

  for letter_sequence in letter_sequence_list: 
    number_sequence = []
    for letter in letter_sequence:

      number = alphabet_dictionary[str(letter)]
      number_sequence.append(number)
    final_sequence.append(np.array(number_sequence))
  return final_sequence

#Function to compute alpha_i(t)
def alpha(i, t, N, T, alpha_matrix, a, b, V): 

  #Check if already computed, if so return
  if alpha_matrix[t][i] != -1:
    return alpha_matrix[t][i]

  #Recursive relation 
  alpha_matrix[t][i] = b[i][V[t]] * sum([alpha(j, t-1, N, T, alpha_matrix, a, b, V) * a[j][i] for j in range(N)])

  return alpha_matrix[t][i]

 
#Function for forwards procedure (populates alpha matrix)
def forwards_procedure(N, T, a, b, initial_prob, V):

   alpha_matrix = -1 * np.ones((T, N))

   #Intialise alpha_i(1) for all i 
   alpha_matrix[0] = initial_prob * b[:, V[0]]

   #First normalisation coefficient
   c_0 = 1/np.sum(alpha_matrix[0, :])

   #Normalise
   alpha_matrix[0] = np.multiply(alpha_matrix[0], c_0)
   normalisation_coefficients = [c_0]

   #Compute alpha for all values of t and i
   for t in range(1, T):
     for i in range(N):
       alpha_matrix[t][i] = alpha(i,t, N, T, alpha_matrix, a, b, V)
     
     if t == (T - 1): 
       likelihood = sum(alpha_matrix[t])

     #Normalise
     c_i = 1/np.sum(alpha_matrix[t, :])
     alpha_matrix[t, :] = np.multiply(alpha_matrix[t, :], c_i)

     #Append normalisation coefficient to use for normalisation for beta values 
     normalisation_coefficients.append(c_i)

   return alpha_matrix, normalisation_coefficients, likelihood 

#Function to compute beta_i(t)
def beta(i,t, N, T, beta_matrix, a, b, V): 

  #Check if already computed and if return it
  if beta_matrix[t][i] != -1:
    return beta_matrix[t][i]

  #Recursive relation
  beta_matrix[t][i] = sum([beta(j, t + 1, N, T, beta_matrix, a, b, V) * a[i][j] * b[j][V[t+1]] for j in range(N)])

  return beta_matrix[t][i]

def backwards_procedure(N, T, a, b, normalisation_coefficients, V):
  beta_matrix = -1 * np.ones((T,N))

  #Intialise beta_i(T) for all i 
  beta_matrix[T - 1] = [1] * N
  #Normalise
  beta_matrix[T - 1] = beta_matrix[T - 1] * normalisation_coefficients[-1]

  #Compute beta values for all t and all i 
  for t in range(T - 2, -1, -1):
    for i in range(N):
      beta_matrix[t][i] = beta(i, t, N, T, beta_matrix, a, b, V)

    #Normalise
    beta_matrix[t] = beta_matrix[t] * normalisation_coefficients[t]

  return beta_matrix

#Function to compute xi_ij(t)
def xi(i, j, t, alpha, beta, a, b, V, N):

  emission = V[t + 1]
  numerator = alpha[t][i] * a[i][j] * beta[t + 1][j] * b[j][emission]
  denominator = sum([alpha[t][k] * a[k][w] * beta[t + 1][w] * b[w][emission] for k in range(N) for w in range(N)])

  return numerator/denominator 

#Function that computes xi_ij(t) for all values of t, i and j 
def populate_xi(N, T, alpha, beta, a, b, V): 
  xi_matrix = np.zeros((N, N, T - 1))
  likelihood = []
  for t in range(T-1):
    A = np.transpose(alpha[t, :])
    B = np.transpose(b[:, V[t + 1]])
    C = beta[t + 1, :]
    denom = np.matmul(np.matmul(A,a) * B , C)
    xi_matrix[:, :, t] = [(alpha[t, i]*a[i, :]* np.transpose(b[:, V[t + 1]]) * np.transpose(C))/ denom for i in range(N)]

  return xi_matrix
 


#Function to return random initial conditions
def return_random_initial_conditions(N_hidden_states, N_emission_characters):
  # Transition probabilities, summing to 1 along each row 
  a = np.zeros((N_hidden_states, N_hidden_states))
  for i in range(N_hidden_states):
    r = [random.random() for i in range(N_hidden_states)]
    s = sum(r)
    r = [ i/s for i in r ]
    a[i] = r

  # Emission probabilities, summing to 1 along each row
  b = np.zeros((N_hidden_states, N_emission_characters))
  for i in range(N_hidden_states):
    r = [random.random() for i in range(N_emission_characters)]
    s = sum(r)
    r = [ i/s for i in r ]
    b[i] = r

  # Initial probabilities, summing to 1
  r = [random.random() for i in range(N_hidden_states)]
  s = sum(r)
  r = [ i/s for i in r ]
  initial_distribution = np.array(r)

  return a, b, initial_distribution




def EM(alphabet, n, S, max_iter = 2000):
    #Convert letters into numbers 
    alphabet_dict = map_alphabet_to_numbers(alphabet)
    list_of_sequences = convert_letter_sequence_to_number_sequence(alphabet_dict, S)
    R = len(S)


    
    a, b, initial_prob = return_random_initial_conditions(n, len(alphabet))

    

    N = a.shape[0]
    old_likelihood = 1 
    likelihood_lst = []
    for it in range(max_iter): # Essentially a while loop as we break out of this one the log likelihood stops increasing
      new_likelihood = 0
      
      #Save these values to update the parameter values at the end of the sequence
      list_of_pi = []
      list_of_a_numerator = []
      list_of_a_denominator = []
      list_of_b_numerator = []
      list_of_b_denominator = []
      for V in list_of_sequences: 
        #Estimation Step
        T = V.shape[0]
        
        alpha_matrix, normalisation_coefficients, likelihood = forwards_procedure(N, T, a, b, initial_prob, V)
        beta_matrix = backwards_procedure(N, T, a, b, normalisation_coefficients, V)
        
        xi = populate_xi(N, T, alpha_matrix, beta_matrix, a, b, V)
        gamma = np.sum(xi, axis=1)

        #Maximization Step
        #Append pi for maximisation step for pi for sequence V
        list_of_pi.append(gamma[:, 0])

        #Append numerator for maximisation step for a for sequence V
        list_of_a_numerator.append(np.sum(xi, axis = 2))
        #Append denominator for maximisation step for a for sequence V
        list_of_a_denominator.append(np.sum(gamma, axis=1, keepdims= True))

        k = np.sum(xi[:,:,T-2],axis=0,keepdims=True).T
        gamma = np.concatenate((gamma, k), axis = 1)
        denominator = np.sum(gamma, axis=1,keepdims = True )

        #Append numerator for maximisation step for b for sequence V
        list_of_b_numerator.append(np.hstack(([np.vstack(np.sum(gamma[:, V == l], axis=1)) for l in range(b.shape[1])])))
        #Append denominator for maximisation step for b for sequence V
        list_of_b_denominator.append(denominator.reshape((-1, 1)))

        #Add log_prob
        log_prob = -np.sum(np.log(normalisation_coefficients))
        new_likelihood += log_prob

      if new_likelihood == old_likelihood:
        break
      else:
        
        old_likelihood = new_likelihood


      #Update parameters using all R sequences 
      initial_prob = sum(list_of_pi)/R
      a = sum(list_of_a_numerator)/sum(list_of_a_denominator)
      b = sum(list_of_b_numerator)/sum(list_of_b_denominator)



    return a, b, initial_prob












#ADDITIONAL CODE SUBMITTED AS EVIDENCE: 

#TESTING CODE 
"""
import random 



for number in range(10): 
  N_hidden_states = 6#random.randint(2,12)
  N_emission_characters = 7#random.randint(2,20)
  a = np.zeros((N_hidden_states, N_hidden_states))
  for i in range(N_hidden_states):
    r = [random.random() for i in range(N_hidden_states)]
    s = sum(r)
    r = [ i/s for i in r ]
    a[i] = r
  # Transition probabilities, summing to 1 along each row 



  b = np.zeros((N_hidden_states, N_emission_characters))
  for i in range(N_hidden_states):
    r = [random.random() for i in range(N_emission_characters)]
    s = sum(r)
    r = [ i/s for i in r ]
    b[i] = r
  # Emission probabilities, summing to 1 along each row


  r = [random.random() for i in range(N_hidden_states)]
  s = sum(r)
  r = [ i/s for i in r ]
  initial_distribution = np.array(r)
  # Initial probabilities, summing to 1


  from hmmlearn import hmm
  #Create model
  model = hmm.MultinomialHMM(n_components=N_hidden_states , init_params="tes")
  model.transmat_ = a
  model.emissionprob_ = b
  model.startprob_ = initial_distribution


  n_sequences = 3#random.randint(0,9)
  list_of_sequences = []

  for i in range(n_sequences):
    n = 10000#random.randint(10,10000)
    visible, hidden = model.sample(n)
    list_of_sequences.append(np.array(visible.reshape(1,-1)[0]))


  def create_random_matrices(N_hidden_states, N_emission_characters): 
    a = np.zeros((N_hidden_states, N_hidden_states))
    for i in range(N_hidden_states):
      r = [random.random() for i in range(N_hidden_states)]
      s = sum(r)
      r = [ i/s for i in r ]
      a[i] = r
    # Transition Probabilities

    b = np.zeros((N_hidden_states, N_emission_characters))
    for i in range(N_hidden_states):
      r = [random.random() for i in range(N_emission_characters)]
      s = sum(r)
      r = [ i/s for i in r ]
      b[i] = r
    # Emission Probabilities

    r = [random.random() for i in range(N_hidden_states)]
    s = sum(r)
    r = [ i/s for i in r ]
    initial_distribution = np.array(r)

    return a, b, initial_distribution

  def create_alphabet(N):
    alphabet = 'abcdefghijklmnoqrstuvwxyz'
    return alphabet[0:N]

  

  def run_BW():
      a, b, pi = create_random_matrices(N_hidden_states, N_emission_characters)

      alphabet = create_alphabet(N_emission_characters)
      input = map_numbers_to_alphabet(list_of_sequences.copy())

      final_a, final_b, final_pi = EM(alphabet, N_hidden_states,input, max_iter = 1000, initial_a = a.copy(), initial_b = b.copy() , initial_distribution = pi.copy())

      lengths = [1] * n_sequences
      model = hmm.MultinomialHMM(n_components=N_hidden_states, n_iter=1000, init_params="", params="ste", tol=0, verbose=False, implementation='scaling')
      model.startprob_ = pi 
      model.transmat_ = a
      model.emissionprob_ = b
      
      model.fit(list_of_sequences, lengths)
      
      diff_a = model.transmat_ -  final_a
      diff_b = model.emissionprob_ - final_b
      diff_pi = model.startprob_ - final_pi 

      biggest_error = max([abs(np.amax(diff_a)), abs(np.amax(diff_b)),abs(np.amax(diff_pi))])
      print('biggest error', biggest_error)
      

  run_BW()


from multiprocessing import Pool
import multiprocessing
from multiprocessing import Process

#process_list = []
#for i in range(1):
#    p =  multiprocessing.Process(target= run_BW)
#    p.start()
#    process_list.append(p)

#for process in process_list:
#    process.join()

"""

#IMPLEMENTATION USING LOGS - Not used as satified with orginal convergence (in comparison with hmmlearn) and it was also slower


"""

import math
import numpy as np 
import random

#Function to map the alphabet to numbers, which is the input our implementation takes
def map_alphabet_to_numbers(alphabet):
  alphabet_dictionary = {}
  count = 0
  for letter in alphabet:
    if letter not in alphabet_dictionary:
      alphabet_dictionary[letter] = count
      count += 1

  return alphabet_dictionary

#Function to convert a sequence of letters to a sequence of numbers using the alphabet_dictionary provided
def convert_letter_sequence_to_number_sequence(alphabet_dictionary, letter_sequence_list):
  final_sequence, number_sequence = [], []
  print(alphabet_dictionary)
  for letter_sequence in letter_sequence_list: 
    for letter in letter_sequence:
      number = alphabet_dictionary[str(letter)]
      number_sequence.append(number)
    final_sequence.append(np.array(number_sequence))
  return final_sequence

#Function to compute alpha_i(t)
def alpha(i, t, N, T, alpha_matrix, a, b, V): 

  #Check if already computed, if so return
  if alpha_matrix[t][i] != np.inf:
    return alpha_matrix[t][i]

  #Recursive relation 
  alpha_matrix[t][i] = b[i][V[t]] + safeLog(sum([safeExp(alpha(j, t-1, N, T, alpha_matrix, a, b, V) + a[j][i]) for j in range(N)]))

  return alpha_matrix[t][i]

 
#Function for forwards procedure (populates alpha matrix)
def forwards_procedure(N, T, a, b, initial_prob, V):

   alpha_matrix = np.ones((T, N)) * np.inf
   #Intialise alpha_i(1) for all i 
   alpha_matrix[0] = initial_prob + b[:, V[0]]
   safeExpV = np.vectorize(safeExp)
   safeLogV = np.vectorize(safeLog)
   c_0 = safeLog(np.sum(safeExpV(alpha_matrix[0])))
   alpha_matrix[0] = np.subtract(alpha_matrix[0], c_0)  

   #First normalisation coefficient
   #c_0 = 1/np.sum(alpha_matrix[0, :])

   #Normalise
   #alpha_matrix[0] = np.multiply(alpha_matrix[0], c_0)
   #normalisation_coefficients = [c_0]
   normalisation_coefficients = [c_0]

   #Compute alpha for all values of t and i
   for t in range(1, T):
     for i in range(N):
       alpha_matrix[t][i] = alpha(i,t, N, T, alpha_matrix, a, b, V)
       
     
     #Normalise
     c_i = safeLog(np.sum(safeExpV(alpha_matrix[t, :])))
     alpha_matrix[t, :] = np.subtract(alpha_matrix[t, :], c_i)
     #c_i = 1/np.sum(alpha_matrix[t, :])
     #alpha_matrix[t, :] = np.multiply(alpha_matrix[t, :], c_i)

     #Append normalisation coefficient to use for normalisation for beta values 
     normalisation_coefficients.append(c_i)

   return alpha_matrix, normalisation_coefficients

#Function to compute beta_i(t)
def beta(i,t, N, T, beta_matrix, a, b, V): 

  #Check if already computed and if return it
  if beta_matrix[t][i] != np.inf:
    return beta_matrix[t][i]

  #Recursive relation
  beta_matrix[t][i] = safeLog(sum([safeExp(beta(j, t + 1, N, T, beta_matrix, a, b, V) + a[i][j] + b[j][V[t+1]]) for j in range(N)]))
            
            

  return beta_matrix[t][i]

def backwards_procedure(N, T, a, b, normalisation_coefficients, V):
  beta_matrix =  np.inf * np.ones((T,N))

  #Intialise beta_i(T) for all i 
  beta_matrix[T - 1] = [safeLog(1.0)] * N
  print(beta_matrix[T - 1])
  #Normalise
  #beta_matrix[T - 1] = beta_matrix[T - 1] * normalisation_coefficients[-1]

  #Compute beta values for all t and all i 
  for t in range(T - 2, -1, -1):
    for i in range(N):
      beta_matrix[t][i] = beta(i, t, N, T, beta_matrix, a, b, V)

    #Normalise
    beta_matrix[t] = beta_matrix[t] - normalisation_coefficients[t]

  return beta_matrix

#Function to compute xi_ij(t)
def xi(i, j, t, alpha, beta, a, b, V, N):

  emission = V[t + 1]
  numerator = alpha[t][i] * a[i][j] * beta[t + 1][j] * b[j][emission]
  denominator = sum([alpha[t][k] * a[k][w] * beta[t + 1][w] * b[w][emission] for k in range(N) for w in range(N)])

  return numerator/denominator 

#Function that computes xi_ij(t) for all values of t, i and j 
def populate_xi(N, T, alpha, beta, a, b, V): 
  xi_matrix = np.zeros((N, N, T - 1))
  gamma_matrix = np.zeros((T-1, N))
  likelihood = []
  safeLogV = np.vectorize(safeLog)
  safeExpV = np.vectorize(safeExp)
  ksis = []
  for t in range(T-1):


    bottom = safeLog(sum([np.sum([safeExpV(alpha[t][i] + a[i, :] + beta[t+1, :] + b[:, V[t+1]])]) for i in range(N)]))
    xi_matrix[:, :, t] =  np.subtract((alpha[t, :] + a[: , :] + beta[t+1, :] + b[:, V[t+1]]),  bottom)   
    gamma_matrix[t, :] = ((alpha[t , :] + beta[t, :]) - safeLogV(sum([safeExpV(alpha[t, :] + beta[t, :])])))   
      
      
    

    

  return xi_matrix, gamma_matrix
 
#Function to check the covergence
def check(model_1, model_2, difference = 0.001):
  model_error = model_1 - model_2
  largest_error = abs(np.amax(model_error))
  if largest_error > difference:
    return True
  return False

#Function to return random initial conditions
def return_random_initial_conditions(N_hidden_states, N_emission_characters):
  # Transition probabilities, summing to 1 along each row 
  a = np.zeros((N_hidden_states, N_hidden_states))
  for i in range(N_hidden_states):
    r = [random.random() for i in range(N_hidden_states)]
    s = sum(r)
    r = [ i/s for i in r ]
    a[i] = r

  # Emission probabilities, summing to 1 along each row
  b = np.zeros((N_hidden_states, N_emission_characters))
  for i in range(N_hidden_states):
    r = [random.random() for i in range(N_emission_characters)]
    s = sum(r)
    r = [ i/s for i in r ]
    b[i] = r

  # Initial probabilities, summing to 1
  r = [random.random() for i in range(N_hidden_states)]
  s = sum(r)
  r = [ i/s for i in r ]
  initial_distribution = np.array(r)

  return a, b, initial_distribution

def convertToLog(lst):
    if (type(lst) is list):
        final = []
        for j in structure:
            final.append(convertToLog(h))
        return final
    return safeLog(structure)

def safeLog(number):
    if (number == 0):
        return -math.inf
    return math.log(number)

def safeExp(number):
    if (number == -math.inf):
        return 0
    return math.exp(number)


#def baum_welch(list_of_sequences, a, b, initial_prob, n_iter=100):

def EM(alphabet, n, S):
    #Convert letters into numbers 
    alphabet_dict = map_alphabet_to_numbers(alphabet)
    list_of_sequences = convert_letter_sequence_to_number_sequence(alphabet_dict, S)
    print(list_of_sequences)
    a, b, initial_prob = return_random_initial_conditions(n, len(alphabet))
    N = a.shape[0]
    R = len(S) # Number of sequences 
    convergence_level = 0.001
    import pandas as pd
    data = pd.read_csv('/content/drive/MyDrive/Baum/data_python.csv.txt')

    V = data['Visible'].values
    list_of_sequences = [V]

    transitions = [[0.5, 0.5], [0.5, 0.5]]
    emissions = [[0.11111111, 0.33333333 ,0.55555556], [0.16666667 ,0.33333333 ,0.5       ]]
    initialDistribution = [0.5, 0.5] #np.array((0.5, 0.5))


    # Convert structures to log space
    lTransitions, lEmissions, lInitialDistribution = convertToLog([transitions, emissions, initialDistribution])
    #print('1', lEmissions)
    lTransitions = np.array(lTransitions)
    lEmissions = np.array(lEmissions, dtype=np.float128)
    #print('2', lEmissions[0][0])
    lInitialDistribution = np.array(lInitialDistribution)

    #probability_of_emission = b
    N = a.shape[0]
    T = V.shape[0]
    for h in range(1):
      #while True: 
      #Save these values to update the parameter values at the end of the sequence
      list_of_pi = []
      list_of_a_numerator = []
      list_of_a_denominator = []
      list_of_b_numerator = []
      list_of_b_denominator = []
      for V in list_of_sequences: 
        #Estimation Step
        T = V.shape[0]
        
        alpha_matrix, normalisation_coefficients = forwards_procedure(N, T, lTransitions, lEmissions,  lInitialDistribution, V)
        #print('A', alpha_matrix)
        beta_matrix = backwards_procedure(N, T, lTransitions, lEmissions, normalisation_coefficients, V)

        #print('sum', sum(alpha_matrix[T-1]))
        #print('B', beta_matrix)
        #print('b', beta_matrix)

        xi = populate_xi(N, T, alpha_matrix, beta_matrix, a, b, V)
        print('XI', xi)
        gamma = np.sum(xi, axis=1)
        #print('likelihood', likelihood)
        #Maximization Step
        #Append pi for maximisation step for pi for sequence V
        list_of_pi.append(gamma[:, 0])

        #Append numerator for maximisation step for a for sequence V
        list_of_a_numerator.append(np.sum(xi, axis = 2))
        #Append denominator for maximisation step for a for sequence V
        list_of_a_denominator.append(np.sum(gamma, axis=1, keepdims= True))

        k = np.sum(xi[:,:,T-2],axis=0,keepdims=True).T
        gamma = np.concatenate((gamma, k), axis = 1)
        denominator = np.sum(gamma, axis=1,keepdims = True )

        #Append numerator for maximisation step for b for sequence V
        list_of_b_numerator.append(np.hstack(([np.vstack(np.sum(gamma[:, V == l], axis=1)) for l in range(b.shape[1])])))
        #Append denominator for maximisation step for b for sequence V
        list_of_b_denominator.append(denominator.reshape((-1, 1)))

      #prev_a = a
      #prev_b = b
      #prev_initial_prob = initial_prob
      #Update parameters using all R sequences 
      initial_prob = sum(list_of_pi)/R
      a = sum(list_of_a_numerator)/sum(list_of_a_denominator)
      b = sum(list_of_b_numerator)/sum(list_of_b_denominator)

      #a_converged_bool = check(a, prev_a, difference = convergence_level)
      #b_converged_bool = check(b, prev_b, difference = convergence_level)
      #initial_prob_converged_bool = check(initial_prob, prev_initial_prob, difference = convergence_level)

      #if (a_converged_bool == False) and (b_converged_bool == False) and (initial_prob_converged_bool == False): 
      #  break



    return a, b, initial_prob

"""





