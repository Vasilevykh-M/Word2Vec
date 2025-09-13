from tqdm import trange

from nltk.stem import SnowballStemmer
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class Prepocessor:
	def __init__(self, language, use_stemming, use_stopwords, window_size):
		self.language = language
		self.use_stemming = use_stemming
		self.use_stopwords = use_stopwords
		self.stop_words = set(stopwords.words(language)) if use_stopwords else set()
		self.stemmer = SnowballStemmer(language) if use_stemming else None
		self.vocabulary = {}
		self.vocab_size = 0
		self.window_size = window_size

	def preprocess_text(self, text):
		tokens = word_tokenize(text.lower(), language=self.language)
		tokens = [token for token in tokens if token.isalpha()]

		if self.use_stopwords:
			tokens = [token for token in tokens if token not in self.stop_words]

		if self.use_stemming:
			tokens = [self.stemmer.stem(token) for token in tokens]
			pos_tags = pos_tag(tokens)
			allowed_pos = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
						   'RB', 'RBR', 'RBS'}
			tokens = [word for word, pos in pos_tags if pos in allowed_pos]

			vocab = set(tokens)
			word_to_idx = {word: i for i, word in enumerate(vocab)}
			idx_to_word = {i: word for i, word in enumerate(vocab)}

			vocab_size = len(vocab)

			print(f"Vocabulary size: {vocab_size}")

			return tokens, word_to_idx, idx_to_word, vocab_size

	def build_vocab_cbow(self, tokens):
		data = []
		for i in trange(self.window_size, len(tokens) - self.window_size, 1, desc="Build vocab"):
			context = [tokens[i - self.window_size], tokens[i - self.window_size + 1],
					   tokens[i + self.window_size - 1], tokens[i + self.window_size]]
			target = tokens[i]
			data.append((context, target))

		return data

	def build_vocab_sg(self, tokens):
		data = []
		for i in range(self.window_size, len(tokens) - self.window_size):
			center_word = tokens[i]
			context_words = [tokens[i - self.window_size], tokens[i - self.window_size + 1],
							 tokens[i + self.window_size - 1], tokens[i + self.window_size]]
			for context_word in context_words:
				data.append((center_word, context_word))

		return data

	def read_data(self, file_path):
		try:
			with open(file_path, 'r') as file:
				content = file.read()
				return content
		except FileNotFoundError:
			print(f"Error: The file '{file_path}' was not found.")
		except Exception as e:
			print(f"An error occurred: {e}")

	def __call__(self, file_path, type_model):
		data = self.read_data(file_path)
		tokens, word_to_idx, idx_to_word, vocab_size = self.preprocess_text(data)
		if type_model == "CBOW":
			data = self.build_vocab_cbow(tokens)

		if type_model == "SG":
			data = self.build_vocab_sg(tokens)
		return data, word_to_idx, idx_to_word, vocab_size