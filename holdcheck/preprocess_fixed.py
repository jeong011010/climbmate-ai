EOF
cat /tmp/preprocess_part1.py >> /tmp/preprocess_fixed_local.py
cat /tmp/preprocess_middle.py >> /tmp/preprocess_fixed_local.py
cat /tmp/preprocess_part2.py >> /tmp/preprocess_fixed_local.py
cat /tmp/preprocess_fixed_local.py | docker exec -i climbmate-app tee /app/holdcheck/preprocess.py > /dev/null
cp /tmp/preprocess_fixed_local.py /app/holdcheck/preprocess.py
echo "Fixed!"
