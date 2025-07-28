download my-python-app.tar file 

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