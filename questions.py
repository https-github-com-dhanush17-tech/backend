import selenium.common.exceptions as exceptions
from selenium import webdriver
import time


def convert(sentence: str):
    driver = webdriver.Chrome()
    driver.get("https://www.lumoslearning.com/llwp/free-question-answer-generator-online.html")
    text_field = driver.find_element_by_id("textarea")
    text_field.send_keys(sentence)
    time.sleep(2)
    button = driver.find_element_by_id('detailsSubmitBtn3')
    driver.execute_script("arguments[0].click();", button)
    while True:
        try:
            question_one = driver.find_element_by_id("question1")
            answer_one = driver.find_element_by_id("answer1")
        except exceptions.NoSuchElementException:
            # Hasn't produced the question yet, so we wait
            continue
        return {
            "Question": question_one.text,
            "Answer": answer_one.text
        }
