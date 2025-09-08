import sys

def main():
    # Print Python version
    print("Python version:", sys.version)

    # Print all args except the script name
    args = sys.argv[1:]
    print("Received arguments:", args)

if __name__ == "__main__":
    main()
