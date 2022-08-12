import operator
from IM import interactionInfo,mi_pairwise
from tqdm import tqdm


class MMRMI:
    
    
    
  
    
    
    def __init__(self):
        pass
        
        


    def select(self, X, y, K, mode = 'pre_eval'):
        ''' select K most informative feature space X according to label space y '''
        
        if mode not in ['pre_eval', 'post_eval']: 
            raise ValueError('invalid mode ==> the mode should be in [pre_eval, post_eval]')
        
        if mode == 'pre_eval': ### issue: what to do if post???
            return self.rank(X, y, mode)[:K]



        
    
    def rank(self, X, y, mode = 'pre_eval'):
        """ Iterate through all the features and find the
        MRMI scores, then find the feature with the highest score, 
        remove it from the feature set and store it in a list.
        Finally return a list named S of the selected features"""



        if mode not in ['pre_eval', 'post_eval']: 
            raise ValueError('invalid mode ==> the mode should be in [pre_eval, post_eval]')
        
        if mode == 'post_eval':
            return self.select(X, y, X.shape[1], mode)


        if mode == 'pre_eval':
            #list of selected features
            S = []
            #list of all feature indices
            F=list(range(X.shape[1]))
        
            
            
            
            #calculate mi for all feature-label pairs and store in a list
            mi_matrix = mi_pairwise(X,y, message= 'Mutual information matrix between features and labels')
            
            #calculate mi for all feature-feature pairs and store in a list
            mi_matrix_features = mi_pairwise(X,X,message= 'Mutual information matrix between features')
            
            #calculate interaction info 
            ii_matrix = interactionInfo(X,X,y,message= 'Interaction information matrix')
            
    
            with tqdm(total=X.shape[1], ncols=80) as t:
                t.set_description('Feature Selection in Progress ')





                #find the feature with the highest MI score
                ##first find sum of all sub lists(sum of mi scores for f_i and L_i)\n",
                ####find max sum and its index( index is a feature)\n",
                ####then add that feature to S and remove from F\n",
            
                list_sum= [sum(sub) for sub in mi_matrix]
                f_max_mi = list_sum.index(max(list_sum))
                S.append(f_max_mi) #add candid feature to S
                F.remove(f_max_mi) # remove feature from F
            



            
                # until no more features are left    
                while F:
                
                    ## for f_i in F'
                    #calculate mrmi score for each feature f_i and/
                    #add f_i index and mrmi score to a list in the form of a tuple
                    list_mrmi_scores=[(f_i,self.MRMI(f_i,S,mi_matrix,ii_matrix,mi_matrix_features,y)) for f_i in F]

                    #find the feature with max mrmi
                    f_max_mrmi = max(list_mrmi_scores , key = operator.itemgetter(1))[0]

                    S.append(f_max_mrmi) #add candid feature  to S       
                    F.remove(f_max_mrmi) # remove selected feature from F
                    t.update(1)

            return S

        
        

        
    def MRMI(self,f_i,S,mi_matrix,ii_matrix,mi_matrix_features, y):
        """ Find the MRMI score for each feature f_i"""
        
        
        
        # answer of second term in mrmi
        list_subtraction = []
        res2 = 0 # sum of I(f_i,f_j,L_i) on the left
        res3 = 0 #sum of I(f_i,f_j,L_i) on the right
        res4 = 0 #sum of I(f_i,L_i) on the right

        
        
        #result of the first addition term
        res1 = 0
        #calculate the mutual info of f_i and l_i = I(f_i,l_i)
        for L_i in range(y.shape[1]):
            res1 = res1 + mi_matrix[f_i][L_i]

            #what to do with this case?
            #if the MI is zero, don't continue to finding f_j
            if res1 == 0:
                return res1
                     

        #for f_j in F-F'
        for f_j in S:
           
            for L_i in range(y.shape[1]):               
                res2 = res2 + ii_matrix[f_i][f_j][L_i]
             
            # if I(f_i,f_j==0) go to next f_j
            if mi_matrix_features[f_i][f_j] ==0:
                continue
            
            res_div1 = res2 / mi_matrix_features[f_i][f_j]
            
            for L_i in range(y.shape[1]):
                if ii_matrix[f_i][f_j][L_i] >= 0:
                    res3 = res3 + ii_matrix[f_i][f_j][L_i]
                          
            for L_i in range(y.shape[1]):
                res4 = res4 + mi_matrix[f_i][L_i]
            res_div2 = res3/res4
            
            res_subtraction = res_div1 - res_div2
            
            list_subtraction.append( res_subtraction)
            # end of second term in mrmi
                   
        #special case
        if list_subtraction == []:
            return (F[0], 0)
        
        #add the score of f_j with max score to the first term
        max_subtraction =  max(list_subtraction )
        final_mrmi = res1 + max_subtraction
        
        return  final_mrmi
