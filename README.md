# HSE_diploma

## How to run

1. Rent server at vast.ai using docker  
	```pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel```
2. Connect to server via 
  ```sh
  ssh -p {port} {user}@{ip} -L 8889:localhost:8888 -L 6007:localhost:6006 
  ```
3. Load code
	```sh
	git clone https://github.com/lizkuh/HSE_diploma.git
	```
4. Install dependencies 
	```sh
	cd HSE_diploma
	./onstart.sh
	'''
5. Run jupyter notebook for experiment
	```sh
	jupyter notebook --allow-root
	```
6. Optional add keys to store experimtnts data to main node, work with git
   Add ssh_keys.zip to host
   unzip ssh_keys.zip
   cp ssh_keys/git* .ssh/
   cp ssh_keys/master* .ssh/
   test it ```ssh -i ~/.ssh/master root@65.124.131.190```
7. Run experiments from ipynb/*.ipynb (it get data from experiments_configs, data, save data to experiments folder that is in .gitignore)

8. To Evaluate model:
    fn=38.txt
    cd /root/HSE_diploma/evaluateFromCodeXGlue/
    python3 calc_code_bleu.py --refs /root/data/reference_corpus.txt --hyp /root/results/$fn --lang java --params 0.25,0.25,0.25,0.25
    python3 calculate_bleu/evaluator.py -a=/root/data/answers.json -p=/root/results/$fn
    
9. For complex evaluation use ipynb/evaluate.ipynb

## Known problems:

- For some model there are no correct EOS token.
	https://github.com/tloen/alpaca-lora/issues/
	https://github.com/huggingface/transformers/pull/22402
	https://github.com/tloen/alpaca-lora/issues/325
   > Crunch fix of it could be done by splitint output to 
   ```res_string.split("## Responce")[-1].split('###')[0]```

- Peft could fail at saving checkpoints for big model
   > It could be change via specific bitsandbytes version
   ```pip install bitsandbytes==0.37.2```