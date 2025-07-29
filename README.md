Option 1 to run docker file

download adobe-challenge.tar file from drive link
https://drive.google.com/drive/folders/1vIVgtqR_HOC0-P7k7oHvDQp3NbkL9cOF?usp=drive_link


open docker on your system

then open path where you save my-python-app.tar in terminal

run this command
docker load -i adobe-hackathon.tar

Make sure the target system has:
input/
output/

then run this command
docker run --rm -v ${PWD}/input_pdfs:/app/input -v ${PWD}/output_jsons:/app/output my-python-app

now the result save in json file save in output_jsons/ directory

option 2

Download entire source code or repo

run these command

cd adobe-hackathon-round1d
docker build -t adobe-challenge .
docker run --rm -v ${PWD}/app/input_pdfs:/app/input -v ${PWD}/app/output_jsons:/app/output adobe-challenge
the output json file save in output directory
