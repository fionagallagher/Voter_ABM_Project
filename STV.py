'''Algorithm to calculate the parliament distribution using a ranked voting method with 
  election quotas and transfered votes (i.e. Single Transferable Vote system)'''
  

import numpy as np

  
def Ranked_vote(number_voters,number_candidates,number_seats,Ideologies_Collection2,quota): 
    '''
    Voters rank each party according to their preference.
    Any candidate(party) that achieve the number of votes required for election - the threshold (or the "quota") 
    are elected and their excess votes are redistributed to the voter's next choice candidate.
    Once this is done, if not all places have been filled then the candidate with the lowest number of votes 
    is eliminated, and their votes are also redistributed to the voter's next choice. 
    
    '''
    threshold=np.round(number_voters/number_seats) 
    Total_seats=number_seats
    parties=list(np.arange(number_candidates))
    Parlaimant_list=np.zeros(5)
    Parlaimant_dict={}                  # Dictionary with elected candidate's index as a  key and seats they got as a value.
    for i in range(number_candidates):
        Parlaimant_dict[i]=0
    
    
    ranked_choice=[]   # Create a ranked candidate choice set  according to each voter's ideology vector.
    for i in range(Ideologies_Collection2.shape[0]):
        temp1=[]
        temp2=np.sort(Ideologies_Collection2[i])[::-1]
        for j in range(number_candidates):
            temp1.append(list(Ideologies_Collection2[i]).index(temp2[j]))
        ranked_choice.append(temp1) 
    ranked_choice1 = np.stack( ranked_choice, axis=0 )  
         
      
    first_counted_votes=[]  # Count voters' first preferences
    for n in parties:
        first_counted_votes.append(list(ranked_choice1[:,0]).count(n))           
    Excess_votes=np.zeros(number_candidates)
    temp_vector=first_counted_votes
    
    while(Total_seats>0):      # This whole process is repeated until all seats are filled. 
        Excess_votes=np.zeros(len(temp_vector))
        '''
         Check which parties have achieved the quota, elect who did into a parliament and count their excess votes.
        '''
        for i in range(len(temp_vector)):
            if temp_vector[i]>=threshold:                
                seats_got=np.round(temp_vector[i]/threshold)   # Candidate i has been elected with the defined seats.
                
                Total_seats-=seats_got             
                if (Total_seats<0):             # If according to votes, candidate gets more seats then it was remained, assign just remaining amount of seats to them.
                    seats_got-=abs(Total_seats)
                    Total_seats+=abs(Total_seats)
                
                Parlaimant_dict[parties[i]]+=seats_got
                Excess_votes[i]+=temp_vector[i]-seats_got*threshold
                
            else:
                Excess_votes[i]+=temp_vector[i]    # If canidate i did not pass the quota, just count it's excess votes.

        if(Total_seats==0 ):
            break       

        """
        Find the party with minimum excess votes and distribute thir votes. 
        """

        mini=np.argmin(Excess_votes)  # Candidate's index, whose votes are being distributed.  
        dist_votes=Excess_votes[mini]   # Number of votes that will be distributed.
        chosen=parties[mini]  # Remove the elected or eliminated party from the remaining candidates' list. 
        parties.remove(parties[mini])   

        if (len(Excess_votes)==2):
            surplus_vector=[dist_votes]
            Excess_votes=list(Excess_votes)
            Excess_votes.remove(Excess_votes[mini])
            Excess_votes[0]+=surplus_vector[0]
            temp_vector=Excess_votes
            continue
        '''
         Count votes for next preferences of voters, who voted for the chosen candidate(elected or eliminated) as a 
         first preference.
        ''' 
        second_counted_votes=[]    
        second_collect_votes=[] 
        
        for i in range(len(ranked_choice1)):
            if ranked_choice1[i][0]==chosen:
                second_collect_votes.append(ranked_choice1[i][1])
        for n in parties:
            second_counted_votes.append(second_collect_votes.count(n))
       
        '''
        Delete the chosen candidate from the original list 
        '''            
        ranked_choice1 = [[ele for ele in sub if ele != chosen] for sub in ranked_choice1]  
        
        '''
        Calculate surplus vector for other candidates, delete the chosen one from the current "excess votes" and redistribute its votes.
        '''
        percentage_vector=np.array(second_counted_votes)/sum(second_counted_votes)     
        surplus_vector=np.round(percentage_vector*dist_votes)          
        Excess_votes=list(Excess_votes)
        Excess_votes.remove(Excess_votes[mini])    
        Excess_votes+=surplus_vector
        temp_vector=Excess_votes
        if(len(Excess_votes)==0):
            break
           
    Parlaimant_list=np.array(list(Parlaimant_dict.values()))/number_seats    # Returns the distribution of Parliament
    return Parlaimant_list