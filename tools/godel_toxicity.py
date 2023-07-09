import random
from tools.toxicity import TC
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
class GODELBlackBoxToxicity:
    def __init__(self, bot_agent, toxicity_obj='answer', tc_name_str='default', random_perturb=False, synonym_method=None, printer=print):
        #### TEST ####
        assert bot_agent in ['GODEL-base','GODEL-large']
        if bot_agent == 'GODEL-base':
            bot_name = 'GODEL-v1_1-base-seq2seq'
        elif bot_agent == 'GODEL-large':
            bot_name = 'GODEL-v1_1-large-seq2seq'
        self.tokenizer = AutoTokenizer.from_pretrained(f"microsoft/{bot_name}", padding_side='left')
        self.model = AutoModelForSeq2SeqLM.from_pretrained(f"microsoft/{bot_name}").cuda()
        self.toxicity_obj = toxicity_obj
        self.tc = TC(tc_name_str=tc_name_str, synonym_method=synonym_method)
        self.random_perturb = random_perturb
        self.num_queries = 0
        self.printer = printer
        assert self.toxicity_obj in ['question', 'answer', 'question_answer'], \
            "toxicity_obj should be in the list ['question','answer','question_answer']."

    def generate(self, question):
        knowledge = ''
        instruction = f'Instruction: given a dialog context, you need to response empathically.'
        dialog = [question]
        dialog = ' EOS '.join(dialog)
        query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
        input_ids = self.tokenizer(f"{query}", return_tensors="pt").input_ids.cuda()
        outputs = self.model.generate(input_ids, max_length=128)
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output

    def get_toxicity_score(self, question, answer, print_op=False):
        if self.toxicity_obj == 'question':
            text = question
        elif self.toxicity_obj == 'answer':
            text = answer
        elif self.toxicity_obj == 'question_answer':
            text  = question.strip() + ' ' + answer.strip()
        if print_op: self.printer(f"============== toxicity ({self.toxicity_obj}) ==============")
        score, text_is_unsafe = self.calc_toxicity(text)
        if print_op: self.printer("{:.6f} ({})".format(score, text_is_unsafe))
        return score, text_is_unsafe

    def is_filter_toxic_question(self, question):
        _, text_is_unsafe = self.calc_toxicity(question)
        return text_is_unsafe

    def calc_toxicity(self, text):
        return self.tc.calc_toxicity(text)

    def calc_toxicity_batch(self, texts):
        """
        Returns the probability that a message is safe according to the classifier.
        """
        return self.tc.calc_toxicity_batch(texts)

    def get_toxicity_score(self, question, answer, print_op=False):
        if self.random_perturb:
            self.printer("we do not implement random perturb in dialogpt")
            raise NotImplementedError
        else:
            if self.toxicity_obj == 'question':
                text = question
            elif self.toxicity_obj == 'answer':
                text = answer
            elif self.toxicity_obj == 'question_answer':
                text  = question.strip() + '\n' + answer.strip()
            score, text_is_unsafe = self.calc_toxicity(text)

        if print_op: self.printer(f"============== toxicity ({self.toxicity_obj}) ==============")
        if print_op: self.printer("{:.6f} ({})".format(score, text_is_unsafe))
        return score, text_is_unsafe

    def is_filter_toxic_question(self, question):
        _, text_is_unsafe = self.calc_toxicity(question)
        return text_is_unsafe
        
    def black_box(self, question, print_op=False, return_answer=False):
        self.num_queries += 1
        if print_op: self.printer("================== Question ===================")
        if print_op: self.printer(question)
        
        if self.toxicity_obj == 'question':
            answer = None
        else:
            answer = self.generate(question)
            if print_op: self.printer("=================== Answer ====================")
            if print_op: self.printer(answer)
        
        score, text_is_unsafe = self.get_toxicity_score(question, answer, print_op=print_op)
        if return_answer:
            return score, text_is_unsafe, answer
        else:
            return score, text_is_unsafe

    def __call__(self, question):
        if type(question) == list:
            # the type of question is AttackedText
            scores, labels = [], []
            for q in question:
                score, label = self.black_box(q.text)
                scores.append(score)
                labels.append(label)
            return scores, labels
        else:
            if type(question) == str:
                return self.black_box(question)
            else:
                # the type of question is AttackedText
                return self.black_box(question.text)

    