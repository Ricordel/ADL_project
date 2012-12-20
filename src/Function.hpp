#include <string>
#include <iostream>


class Function
{
        public:
                virtual ~Function();

                /* This produces the canonical form required for the project,
                 * for example 1,2,(5,6) */
                virtual std::string toString() const = 0;

                /* This produces a nicely readable string representation of the
                 * functions, for example x_1 + x_2 + x_5.x_6 */
                virtual std::string toPrettyString() const = 0;

                /* Return the length of the NLFSR cycle for this particular function */
                inline uint32_t getCycleLength() const {
                        int length = 0;
                        while (this->nextVal() != startVal) {
                                length++;
                        }

                        return length;
                }

        protected:
                uint32_t curVal;
                uint32_t startVal = 1;
                
                /* Side effects on the curVal member AND returns its value */
                virtual uint32_t nextVal() const = 0;
};




class Function_a_b_cd : public Function
{
        public:
                Function_a_b_cd();
                Function_a_b_cd(uint32_t a, uint32_t b, uint32_t c, uint32_t d);
                virtual ~Function_a_b_cd();

                virtual std::string toString() const;
                virtual std::string toPrettyString() const;
        
        private:
                uint32_t m_a;
                uint32_t m_b;
                uint32_t m_c;
                uint32_t m_d;

        protected:
                virtual inline uint32_t nextVal() const {
                        std::cout << "Mettre la génération à la Elena ici" << std::endl;
                        return 0;
                }
};
