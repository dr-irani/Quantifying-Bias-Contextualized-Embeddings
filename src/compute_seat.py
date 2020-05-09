from transformers import BertModel, BertConfig


class Embedding(object):
    """
    Extend this class and overide get_embedding method with whatever is used to
    generate the embedding
    """
    def get_embedding(self, sentence):
        pass

class BERTEmbedding(Embedding):

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def get_embedding(self, sentence, max_len=64):
        """
        Input formtting for BERT
        """
        encoded_dict = tokenizer.encode_plus(
            sentence, 
            add_special_tokens = True,    # Add '[CLS]' and '[SEP]'
            max_length = max_len,         # Pad & truncate all sentences.
            pad_to_max_length = True,
            return_attention_mask = True, # Construct attn. masks.
            return_tensors = 'pt',        # Return pytorch tensors.
        )
        
        """
        Creating word/sentence vectors
        """
        # Predict hidden states features for each layer
        with torch.no_grad():
          hidden_states = self.model(encoded_dict['input_ids'])[2]

        token_embeddings = torch.stack(hidden_states, dim=0)   # (12, batch_size, token_length, embedding_size)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1,0,2)  # (token_length, 12, embedding_size (768))

        """
        Sentence vectors
        """
        token_vecs = hidden_states[11][0]
        sentence_embedding = torch.mean(token_vecs, dim=0)

        return sentence_embedding


class SEAT(object):

    def __init__(self, model_directory):
        config = BertConfig.from_pretrained(model_directory, output_hidden_states=True)
        self.model = BertModel.from_pretrained(model_directory, config=config)
        self.tokenizer = BertTokenizer.from_pretrained(model_directory)
        self.model.eval()
        self.embedding = BERTEmbedding(self.model, self.tokenizer)


    def cossim(self, x, y):
        return np.dot(x, y) / math.sqrt(np.dot(x, x) * np.dot(y, y))


    def cossim_matrix(self, attr, targ):
        nrows = len(targ)
        ncols = len(attr)
        m = np.zeros((nrows, ncols))
        for i in range(nrows):
            for j in range(ncols):
                m[i, j] = self.cossim(targ[i], attr[j])

        return m


    def calculate_seat_score(self, m, targ_size, attr_size):
      
        # matrix of size target that holds s_wAB for every target sentence
        s_wAB = np.mean(m[:, :attr_size], axis=1) - np.mean(m[:, attr_size:], axis=1)

        seat_score = np.sum(s_wAB[:targ_size]) - np.sum(s_wAB[targ_size:])
        effect_size = (np.mean(s_wAB[:targ_size]) - np.mean(s_wAB[targ_size:])) / np.std(s_wAB, ddof = 1)
        return seat_score, effect_size


    def run_seat(self, A, B, X, Y):
        attr = {}
        targ = {}

        for i in range(len(A)):
            attr[i] = self.embedding.get_embedding(A[i])
            attr[i+len(A)] = self.embedding.get_embedding(B[i])

        for i in range(len(X)):
            targ[i] = self.embedding.get_embedding(X[i])
            targ[i+len(X)] = self.embedding.get_embedding(Y[i])
      
        m = self.cossim_matrix(attr, targ)
        seat_score, effect_size = self.calculate_seat_score(m, len(X), len(A))

        return m, seat_score, effect_size


def create_target_attribute_lists(white_names, black_names):
    """
    Creating list of attributes and target words
    attributes: A - fear, B - joy
    target: X - white names, Y - black names
    """
    A_words = ['fear', 'scared', 'dreadful', 'horrible', 'terrified', 'anxious', 'discouraged', 'threatened', 'shocked']
    B_words = ['joy', 'great', 'wonderful', 'amazing', 'happy', 'excited', 'glad', 'ecstatic', 'relieved']


    # Creating sentences and converting them to embeddings
    X, Y = [], []

    races = ['white', 'black']
    for i, race in enumerate(races):
        if i == 0:
            X.append("This person is " + race)
            X.append("This is a " + race + " person")
            X.append("A " + race + " person is here")
        else:
            Y.append("This person is " + race)
            Y.append("This is a " + race + " person")
            Y.append("A " + race + " person is here")
      
    A = []
    B = []
    for emotion in A_words:
        A.append("They make me feel " + emotion)
        A.append("They feel " + emotion)
        A.append("The situation makes them feel " + emotion)
    
    for emotion in B_words:
        B.append("They make me feel " + emotion)
        B.append("They feel " + emotion)
        B.append("The situation makes them feel " + emotion) 

    assert(len(X) == len(Y))
    assert(len(A) == len(B))
    return A, B, X, Y

