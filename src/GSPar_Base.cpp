
#include <chrono>
#include <string>
#include <algorithm> //std::generate_n

namespace GSPar {
    static bool srandInitiated = false;

    std::string getRandomString(short length) {
        if (!srandInitiated) {
            // Initialize random seed with ms since linux epoch
            std::srand(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
            srandInitiated = true;
        }

        auto randchar = []() -> char {
            const char charset[] =
            "0123456789"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz";
            const size_t max_index = (sizeof(charset) - 1);
            return charset[ std::rand() % max_index ];
        };
        std::string generatedName(length,0);
        std::generate_n(generatedName.begin(), length, randchar);
        return generatedName;
    }
}
