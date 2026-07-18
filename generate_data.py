from transformers import AutoTokenizer
import argparse
import json

# python generate_data.py --input_len 100 --num_rows 500 > 100-500.jsonl
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_len", type=int, required=True)
    parser.add_argument("--num_rows", type=int, default=1)
    parser.add_argument("--base_req", type=str, default="Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("/mnt/share/l00850654/weights/dsv3_w8a8")
    num_rows = args.num_rows
    input_len = args.input_len
    s = [args.base_req]

    tokens = tokenizer.encode(s[0])
    #print(f"base token length is {len(tokens)}")
    
    ss = s * (input_len // len(tokens))
    sss = " ".join(ss)

    d = {"question":sss, "answer":"First calculate the cost of the movie tickets by multiplying the price per ticket by 2: $7.50 * 2 = $<<7.5*2=15.00>>15.00\nThen add up all the money Janet spent: $3.50 + $15.00 + $8.50 = $<<3.5+15+8.5=27.00>>27.00\nNow subtract Janet's spending from the initial amount of money she had: $40.00 - $27.00 = $<<40-27=13.00>>13.00\n#### 13"}
    json_d = json.dumps(d)

    res = ""

    for i in range(num_rows-1):
        res += json_d + "\n"
    res += json_d
    print(res)
