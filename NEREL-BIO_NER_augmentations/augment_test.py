from nlpaug.augmenter.word import SynonymAug
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

aug = SynonymAug(aug_src='wordnet', aug_p=0.5)
print(aug.augment("The doctor prescribed medication for the patient."))
