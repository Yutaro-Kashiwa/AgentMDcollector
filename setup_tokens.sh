#!/bin/bash

# Setup script for multiple GitHub tokens
# This helps you configure multiple tokens for parallel collection

echo "======================================"
echo "GitHub Multi-Token Setup Helper"
echo "======================================"
echo ""

# Function to test token
test_token() {
    local token=$1
    local response=$(curl -s -H "Authorization: token $token" https://api.github.com/rate_limit)
    
    if echo "$response" | grep -q "rate"; then
        local remaining=$(echo "$response" | grep -o '"search":{[^}]*' | grep -o '"remaining":[0-9]*' | cut -d: -f2)
        echo "✅ Valid (Search API: $remaining/30 remaining)"
        return 0
    else
        echo "❌ Invalid token"
        return 1
    fi
}

# Check for existing tokens
echo "Checking for existing tokens..."
echo ""

existing_count=0

# Check main token
if [ ! -z "$GITHUB_TOKEN" ]; then
    echo -n "GITHUB_TOKEN: "
    test_token "$GITHUB_TOKEN"
    ((existing_count++))
fi

# Check numbered tokens
for i in {1..20}; do
    var_name="GITHUB_TOKEN_$i"
    token="${!var_name}"
    
    if [ ! -z "$token" ]; then
        echo -n "$var_name: "
        test_token "$token"
        ((existing_count++))
    fi
done

echo ""
echo "Found $existing_count existing token(s)"
echo ""

# Ask to add more tokens
read -p "Do you want to add more tokens? (y/n): " add_more

if [ "$add_more" = "y" ] || [ "$add_more" = "Y" ]; then
    echo ""
    echo "You can create new tokens at: https://github.com/settings/tokens"
    echo "Required scope: public_repo (for public repositories)"
    echo ""
    
    read -p "How many tokens do you want to add? " num_tokens
    
    # Create tokens file
    touch tokens.txt
    
    for ((i=1; i<=num_tokens; i++)); do
        echo ""
        read -s -p "Enter token $i (hidden): " token
        echo ""
        
        if [ ! -z "$token" ]; then
            echo -n "Testing token $i... "
            if test_token "$token"; then
                echo "$token" >> tokens.txt
                
                # Also create export commands
                next_num=$((existing_count + i))
                echo "export GITHUB_TOKEN_$next_num='$token'" >> setup_env.sh
            fi
        fi
    done
    
    echo ""
    echo "✅ Tokens saved to tokens.txt"
    echo "✅ Environment exports saved to setup_env.sh"
    echo ""
    echo "To use these tokens:"
    echo "  1. Source the environment: source setup_env.sh"
    echo "  2. Or the script will read from tokens.txt automatically"
fi

# Create example configuration
cat > token_config.json << EOF
{
    "collection_strategy": "parallel",
    "max_workers_per_token": 1,
    "delay_between_requests": 1,
    "auto_rotate": true,
    "save_interval": 60,
    "target_files": [
        "Claude.md", "Agent.md", "openai.md", "AI.md", "LLM.md",
        ".cursorrules", "prompts.json", "claude.json", "agent.json"
    ]
}
EOF

echo ""
echo "Configuration saved to token_config.json"
echo ""
echo "To start collection with all tokens:"
echo "  python multi_token_collector.py"
echo ""
echo "======================================"
echo "Setup complete!"
echo "======================================