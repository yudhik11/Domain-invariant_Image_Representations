from misc import formKernel
import numpy as np

def GetConstraints(y1, y2):
    '''
    Input :: 
        y1(list) : Labels available for the data
        y2(list) : Labels for training data
    Output :: 
        C(np.ndarray) : Constraint matrix built for the data
    '''
    pos=0;
    ly1=len(y1);
    ly2=len(y2);
    C=np.zeros((ly1*ly2,4));
    for i in range(ly1):
        for j in range(ly2):
            if(y1[i]==y2[j]):
                C[pos,:]=[i, j+ly1, 1, -1]; # w'Ax > 1; -w'Ax < -1;
            else:
                C[pos,:]=[i, j+ly1, -1, 1]; # w'Ax < -1
            pos=pos+1
    return C

def learnAsymmTransformWithSVM(XA, yA, XB, yB, params):
    dA = XA.shape[1];
    dB = XB.shape[1]
    
    # Get constraints for data
    C = GetConstraints(yA,yB);


    X = np.vstack((XA, XB))

    # Get normalised data
    K0train = formKernel(X, X, params)

    # Function to perform frobenius based transformation learning of parameters
    S,_,_ = asymmetricFrob_slack_kernel(K0train,C,params['gamma'],10e-3);

    L = np.eye(dA) + np.matmul(X.T, np.matmul(S, X));
    return L

def asymmetricFrob_slack_kernel(KA, C, gamma = None,thresh = 10e-3):
    if thresh is None:
        thresh=10e-3;

    if gamma is None:
        gamma = 1e1;

    maxit = 1e4;
    
    [nA,nA] = KA.shape
    S = np.zeros((nA,nA))
    [c,t] = C.shape
    slack = np.zeros((c,1));
    lambda1 = np.zeros((c,1));
    lambda2 = np.zeros((c,1));
    v = (C[:, 0] *nA) + C[:, 1]
    v = np.array(v, dtype = np.int)
    mf1= (KA.flatten(1)[0, v] - C[:, 2])
    mf2= C[:,3].reshape(1, -1)
    viol = np.multiply(mf1,mf2)
    viol = viol.T;
    for i in range(int(maxit)):
        curri = np.argmax(viol)
        mx = np.max(viol)
        if mx < thresh*1000:
            break;
    
        p1 = int(C[curri,0])
        p2 = int(C[curri,1])
        b = int(C[curri,2])
        s = int(C[curri,3])
        kx = KA[p1,:];
        ky = KA[:,p2];
        
        arg1 = lambda1[curri]
        arg2 = np.matmul(kx, np.matmul(S,ky)) 
        arg3 = (1/gamma + KA[p1,p1]*KA[p2,p2])
        print(arg1, arg2, arg3)
        
        arg4 = s * (b - KA[p1, p2]- arg2 -slack[curri])
        alpha = min(arg1, arg4 / arg3)
        
        
        lambda1[curri] = lambda1[curri] - alpha;
        S[p1,p2] = S[p1,p2] + s*alpha[0,0];
        
        slack[curri] = slack[curri] - alpha/gamma;

        alpha2 =  min(lambda2[curri],gamma*slack[curri]);
        slack[curri] = slack[curri] - alpha2/gamma;
        lambda2[curri] = lambda2[curri] - alpha2;

#         update viols
        C = np.array(C, dtype = np.int)
        v = KA[C[:,0],p1]
        w = KA[p2,C[:,1].T]
        
        arg1 = np.multiply(KA[C[:,0],p1].T, KA[p2,C[:,1]])
        arg2 = alpha[0,0]*np.multiply(C[:,3].T, arg1)
        viol = viol + (s*arg2).T
        
        viol[curri] = viol[curri] + (alpha+alpha2)/gamma
        
    return [S, slack, i]