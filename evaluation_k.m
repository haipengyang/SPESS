function [population,popSize,fitness] = evaluation_k(offspring,population,fitness,popSize,k,X)
        offspringFit    = zeros(1,2);
        offspringFit(2) = sum(offspring);    
        if offspringFit(2)>0 && offspringFit(2)<k+1           
            pos = offspring==1;
            offspringFit(1) = norm(X(:,pos)*pinv(X(:,pos))*X,'fro')^2;           
            if ~(sum((fitness(1:popSize,1)>offspringFit(1)).*(fitness(1:popSize,2)<=offspringFit(2)))+sum((fitness(1:popSize,1)>=offspringFit(1)).*(fitness(1:popSize,2)<offspringFit(2)))>0)
                deleteIndex = ((fitness(1:popSize,1)<=offspringFit(1)).*(fitness(1:popSize,2)>=offspringFit(2)))'; 
                ndelete     = find(deleteIndex==0);
                population  = [population(ndelete,:);offspring];
                fitness     = [fitness(ndelete,:);offspringFit];          
                popSize     = length(ndelete)+1;
            end
        end   
end