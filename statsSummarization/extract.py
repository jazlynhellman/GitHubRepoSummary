import os
import json
import mistune
import re
import pandas as pd
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter

from nltk.tokenize.treebank import TreebankWordDetokenizer


from anytree import Node, RenderTree

from sklearn.feature_extraction.text import TfidfVectorizer 
tfidf_vectorizer=TfidfVectorizer(use_idf=True)

from langdetect import detect



def find_url(str): # get urls starting with www.
    return list(set(re.findall('www.(?:[-\w.]|(?:%[\da-fA-F]{2}))+', str)))


def jaccard_sim(dir_path, link): # calculate jaccard similarity score on dir_path and weblinks
    dir_path = set(dir_path)
    link = set(link)
    intersection = dir_path.intersection(link)
    return float(len(intersection)) / (len(dir_path) + len(link) - len(intersection))


def keep_alpha(str): # keep only alphabets
	return ''.join(x for x in str if x.isalpha()).lower()


def sort_by_score(score, urls): # sort by another list
	return [x for _,x in sorted(zip(score,urls))]


def case_parser(paragraph): #camelCase PascalCase parser
	cameled = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', paragraph)
	uncameled = re.sub('([a-z0-9])([A-Z])', r'\1 \2', cameled).lower()
	return uncameled


def parse_text_code(paragraph): # separates code block with natural language
	paragraph = paragraph.replace("```", "@CODE@")

	code_block_out = []
	code_block = re.findall(r'@CODE@.+?@CODE@',paragraph)

	for b in code_block:
		paragraph = paragraph.replace(b, '')
		code_block_out.append(b.replace('@CODE@',''))

	cleaned_paragraph = re.sub(r'[^a-zA-Z ]+', ' ', paragraph)
	cleaned_paragraph = [w for w in word_tokenize(cleaned_paragraph.lower()) if len(w) < 25]
	return cleaned_paragraph, code_block_out


def compute_tfidf(docs):
	tfidf=tfidf_vectorizer.fit_transform(docs)
	# tfidf by document: tfidf[DOC_IDX]
	# get vocab idx : tfidf.vocabulary_

	tfidf_dict = dict(zip(tfidf_vectorizer.get_feature_names(), tfidf_vectorizer.idf_))
	return tfidf_dict

	"""
	# tf-idf values in a pandas data frame by doc
	df = pd.DataFrame(ifidf[DOX_IDX].T.todense(), index=tfidf.get_feature_names(), columns=["tfidf_score"])
	df.sort_values(by=["tfidf_score"],ascending=False)
	"""

	#camelCase PascalCase snake_case kebab-case



def main():

	top_folder = os.listdir('./')
	project_counter = 0
	empty_readme_counter = 0
	all_urls = {}
	all_body = []

	for item in top_folder:

		project_name = item
		txt_to_json = {}
		file_structure = []
		file_tree = Node(project_name)

		if os.path.isdir(item): # only walk through folders
			for dirpath, dirs, files in os.walk(item):
				print(type(dirpath), type(dirs), type(files))
				print(type(dirpath), type(dirs), type(files))

				file_structure_sub = []


				# discard hidden files
				files = [f for f in files if not f[0] == '.']
				dirs[:] = [d for d in dirs if not d[0] == '.']
				

				for f in files:
					if f.lower().endswith('.md'): # only look at .md files
					

				#Extract strs from markdown
						with open(dirpath+"/"+f, encoding = "ISO-8859-1") as md_file:
							file_split = [line.split() for line in md_file]
							file_split_flatten = [val for sublist in file_split for val in sublist]
							md_file.close()

							joined_paragraph = " ".join(file_split_flatten)
							joined_paragraph = case_parser(joined_paragraph) # parsing camel and pascal cases
							tokenized_paragraph, code_block = parse_text_code(joined_paragraph)
							if len(tokenized_paragraph) < 20:
								empty_readme_counter += 1
							all_body.append(project_name + f + ' '.join(tokenized_paragraph))

							
							## Detect Language of all md files ##
							## use the tokenized_paragraph param to capture .md's without code
							untokenized_paragraph = TreebankWordDetokenizer().detokenize(tokenized_paragraph)
							lang = detect(joined_paragraph)

							


						txt_to_json.update({'project_name': project_name})
						txt_to_json.update({'file_name':f})
						txt_to_json.update({'joined_body':joined_paragraph})
						txt_to_json.update({'tokenized_body':tokenized_paragraph})
						txt_to_json.update({'code':code_block})

						txt_to_json.update({'natural_language':lang})


				# Extract URLs
						for i in sent_tokenize(joined_paragraph):
							urls = find_url(i)
							if len(urls) > 0:
								if len(urls) > 2: # if more than 3 weblinks fetched, calculate similarity score
									similarity_score = []
									for i in urls:
										weblink = i[4:].split('.', 1)[0]
										similarity_score.append(jaccard_sim(dirpath, weblink))
									sorted_urls = sort_by_score(similarity_score, urls)
									all_urls.update({project_name: sorted_urls[-3:]})
									txt_to_json.update({'websites': (project_name, sorted_urls[-3:])})
								else:
									all_urls.update({project_name: urls})
									txt_to_json.update({'websites': (project_name, urls)})


				# get file structure
					## folders

				path = dirpath.split('/')
				path_tree = Node(path)

				path_name = keep_alpha(os.path.basename(dirpath))
				folder = (len(path), path_name, 'folder')
				file_structure_sub.append(folder)
				this_folder = Node(folder, parent = path_tree)
				this_path = Node(folder, parent = file_tree)

					## files
				for f in files:
					file_name_ext = os.path.splitext(f)
					file_name = keep_alpha(file_name_ext[0])
					file_ext = case_parser(file_name_ext[1][1:])
					if file_name and file_ext:
						file = (len(path), file_name, file_ext)
						file_structure_sub.append(file)
				file_structure.append(file_structure_sub)
			txt_to_json.update({'file_structure': file_structure})
			#print(txt_to_json)



			for pre, fill, node in RenderTree(file_tree):
			    print("%s%s" % (pre, node.name))

			exit()
			# save a pickle file for every project
			project_counter += 1
			pickle_save = open((project_name + "/out.pickle"),"wb")
			pickle.dump(txt_to_json, pickle_save)
			pickle_save.close()


	print("Number of project in this batch: ", project_counter)
	print("Number of websites in this batch: ", len(all_urls))
	print("Number of empty READMEs in this batch: ", empty_readme_counter)

	#print(compute_tfidf(all_body))


	# Extracting URLS for web scrapping
	pickle_save = open("all_urls.pickle","wb")
	pickle.dump(all_urls, pickle_save)
	pickle_save.close()

	"""
	# TO LOAD PICKLE FILE
	pickle_load = open("all_urls.pickle","rb")
	loaded = pickle.load(pickle_load)
	print(loaded)
	"""


if __name__ == '__main__':


    main()

