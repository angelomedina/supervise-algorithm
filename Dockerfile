FROM python:3

ADD regresionLineal.py /
ADD covid19CR.csv /

RUN pip install numpy
RUN pip install matplotlib
RUN pip install pandas
RUN pip install sklearn

CMD [ "python", "./regresionLineal.py" ]