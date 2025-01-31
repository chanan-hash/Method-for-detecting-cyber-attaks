#include <iostream>

int sum(std::string s) {
    int result = 0, num = 0;

    for (char ch : s) {
        if (ch == '+') {
            result += num; // Add the previous number to the result
            num = 0;        // Reset num for the next number
        } else {
            num = num * 10 + (ch - '0'); // Build the number digit by digit
        }
    }
    
    result += num; // Add the last number

    return result;
}

int main() {
    std::cout << sum("1+2+12") << std::endl; // Output: 15
    std::cout << sum("34") << std::endl;     // Output: 34
    std::cout << sum("10+20+30+40") << std::endl; // Output: 100
    std::cout << sum("10+20+30+40+50+9+1+7+8+5+20") << std::endl; // Output: 200
    return 0;
}