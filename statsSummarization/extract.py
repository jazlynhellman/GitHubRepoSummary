import os
import json
import mistune
import re
import pandas as pd
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer 
tfidf_vectorizer=TfidfVectorizer(use_idf=True)




def find_url(str): # get urls starting with www.
    return list(set(re.findall('www.(?:[-\w.]|(?:%[\da-fA-F]{2}))+', str)))


def jaccard_sim(dir_path, link): # calculate jaccard similarity score on dir_path and weblinks
    dir_path = set(dir_path.split()) 
    link = set(link.split())
    intersection = dir_path.intersection(link)
    return float(len(intersection)) / (len(dir_path) + len(link) - len(intersection))


def keep_alpha(str):
	return ''.join(x for x in str if x.isalpha()).lower()


def sort_by_score(score, urls):
	return [x for _,x in sorted(zip(score,urls))]


def parse_text_code(paragraph):
	paragraph = paragraph.replace("```", "@CODE@")

	code_block_out = []
	code_block = re.findall(r'@CODE@.+?@CODE@',paragraph)

	for b in code_block:
		paragraph = paragraph.replace(b, '')
		code_block_out.append(b.replace('@CODE@',''))

	cleaned_paragraph = re.sub(r'[^a-zA-Z ]+', '', paragraph)
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



top_folder = os.listdir('./')
project_counter = 0

all_urls = {}
all_body = []

for path in top_folder:

	project_name = path
	txt_to_json = {}
	file_structure = []

	if os.path.isdir(path): # only walk through folders
		project_counter += 1
		for dirpath, dirs, files in os.walk(path):
			file_structure_sub = []


			# discard hidden files
			files = [f for f in files if not f[0] == '.']
			dirs[:] = [d for d in dirs if not d[0] == '.']
			

			for f in files:
				if f.lower().endswith('.md'): # only look at .md files
				

			#Extract strs from markdown
					with open(dirpath+"/"+f) as md_file:
						file_split = [line.split() for line in md_file]
						file_split_flatten = [val for sublist in file_split for val in sublist]
						md_file.close()

						joined_paragraph = " ".join(file_split_flatten)
						all_body.append(joined_paragraph)
						tokenized_paragraph, code_block = parse_text_code(joined_paragraph)

					txt_to_json.update({'file_name':f})
					txt_to_json.update({'tokenized_body':joined_paragraph})
					txt_to_json.update({'tokenized_body':tokenized_paragraph})
					txt_to_json.update({'code':code_block})


			# Extract URLs
					for i in sent_tokenize(joined_paragraph):
						urls = find_url(i)
						if len(urls) > 0:
							if len(urls) > 3: # if more than 3 weblinks fetched, calculate similarity score
								similarity_score = []
								for i in urls:
									similarity_score.append(jaccard_sim(dirpath, i))
								sorted_urls = sort_by_score(similarity_score, urls)
								all_urls.update({project_name: sorted_urls[-3:]})
								txt_to_json.update({'websites': (project_name, sorted_urls[-3:])})
							else:
								all_urls.update({project_name: urls})
								txt_to_json.update({'websites': (project_name, urls)})


			# get file structure
				## folders
			path = dirpath.split('/')
			path_name = keep_alpha(os.path.basename(dirpath))
			folder = (len(path), path_name, 'folder')
			file_structure_sub.append(folder)

				## files
			for f in files:
				file_name_ext = os.path.splitext(f)
				file_name = keep_alpha(file_name_ext[0])
				file_ext = file_name_ext[1][1:].lower()
				if file_name:
					file = (len(path), file_name, file_ext)
					file_structure_sub.append(file)
			file_structure.append(file_structure_sub)
		txt_to_json.update({'file_structure': file_structure})
		#print(txt_to_json)



		# save a pickle file for every project
		pickle_save = open((path[0] + "/out.pickle"),"wb")
		pickle.dump(txt_to_json, pickle_save)
		pickle_save.close()

print("Number of project in this batch: ", project_counter)
print("Number of websites in this batch: ", len(all_urls))
print(compute_tfidf(all_body))


"""
# Extracting URLS for web scrapping
pickle_save = open("all_urls.pickle","wb")
pickle.dump(all_urls, pickle_save)
pickle_save.close()
"""



"""
# TO LOAD PICKLE FILE
pickle_load = open(FILE_NAME,"rb")
loaded = pickle.load(pickle_load)
print(loaded)
"""
