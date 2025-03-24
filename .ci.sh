sudo apt update -y
sudo apt install -y jq

mkdir tmp

echo -n "云南沃森生物技术股份有限公司成立于 2001 年,经过二十余年的发展,现已成为国内专业从事人用疫苗等生物技术药物集研发、生产、销售于一体的现代高技术生物制药企业。" >tmp/doc.md

nohup python rag-serve.py tmp 2>&1 &
sleep 20

curl -v -H 'Knowledge-Model: openai' -H 'Knowledge-Model-Name: deepseek-chat' -H "Knowledge-Url: https://api.deepseek.com" -F 'file=@tmp/doc.md' 127.0.0.1:8080/index/org/agent
curl -v -d '{"turns":[{"role": "user", "content" : "沃森生物?"}]}' 127.0.0.1:8080/query/org/agent | tee /dev/stderr | jq .reply | grep "云南沃森生物技术股份有限公司" || exit 1
sleep 1
