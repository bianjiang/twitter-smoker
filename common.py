#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

import re, os, csv, json
from ftfy import fix_text

twitter_username_p = re.compile(r'(?<=^|(?<=[^a-zA-Z0-9-_]))(@[A-Za-z]+[A-Za-z0-9_]+)')
url_p = re.compile(r'http[s]?:.*?(\s+|$)')
hashtag_p = re.compile(r'#\w+')
email_p = re.compile(r'(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)')
nonvalid_characters_p = re.compile("[^a-zA-Z0-9#\*\-_\s]")
non_ascii_p = re.compile(r'[^\x00-\x7F]+')


def clean_tweet_text(text):
    text = fix_text(text.replace('\r\n',' ').replace('\n',' ').replace('\r',' '))

    text = re.sub(non_ascii_p, '', text)

    return text.strip()

def remove_url(text):
    text = re.sub(url_p, '', text)

    return text

def remove_username(text):
    text = re.sub(twitter_username_p, '', text)

    return text

def alpha_and_number_only(text):

    text = re.sub(nonvalid_characters_p, ' ', text)
    #text = text.replace('-', ' ')

    return text

def sanitize_text(text):

    return alpha_and_number_only(remove_username(remove_url(clean_tweet_text(text))))

def has_url(text):

    m = re.findall(url_p, text)

    return 1 if m else 0

# text = 'RT @JHSty: @BillSchulz Try this: Smokey Robinson & The Miracles-The Tears Of A Clown http://bit.ly/EFjjb  Hang in...Great Song!!!'
# logger.info(has_url(text))

# quit()

def has_username(text):

    m = re.findall(twitter_username_p, text)

    return 1 if m else 0

parse_time_format = '%a %b %d %H:%M:%S +0000 %Y'
day_output_date_format = '%Y%m%d_%a'
month_output_date_format = '%Y%m'
import time

def time_str_to_day(time_str):

    t = time.strptime(time_str, parse_time_format)

    return time.strftime(day_output_date_format, t)

def time_str_to_month(time_str):

    t = time.strptime(time_str, parse_time_format)

    return time.strftime(month_output_date_format, t)


def get_fieldnames(csv_filename):
    fieldnames = []
    with open(csv_filename, 'r', newline='', encoding='utf-8') as rf:
        reader = csv.reader(rf)

        header = next(reader)

        for field in header:
            fieldnames.append(field)

    return fieldnames

username_blacklist = []

def get_hashtag(term):
    if term.startswith('#'):
        return None
    elif (re.findall(r'[^a-zA-Z0-9\-]', term)): # if the term has characters that are not allowed in hashtags
        return None
    else:
        return '#%s'%(term)

