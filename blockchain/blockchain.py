import hashlib
import json
from time import time
from uuid import uuid4
from textwrap import dedent

import requests

from flask import Flask, jsonify, request

# sourced from: https://hackernoon.com/learn-blockchains-by-building-one-117428612f46

class Blockchain(object):
    def __init__(self):
        self.chain = []
        self.current_transactions = []

        self.new_block(previous_hash=1, proof=100)

    def new_block(self, proof, previous_hash = None):
        """
        Create a new block in the chain
        :param proof: <int> Proof given by the proof of work algo
        :param previous_hash: (optional) <str> Hash of prev block
        :return: <dict> new block
        """

        block = {
            'index': len(self.chain) + 1,
            'timestamp': time(),
            'transactions': self.current_transactions,
            'proof': proof,
            'previous_hash': previous_hash or self.hash(self.chain[-1])
        }

        self.current_transactions = []

        self.chain.append(block)
        return block

    def new_transaction(self, sender, recipient, amount):
        """
        Creates a new transaction to go into the next mined block.

        :param sender: <str> Address of the Sender
        :param recipient: <str> Address of the recipient
        :param amount: <int> amount
        :return: <int> Index of the block that will hold this transaction
        """

        self.current_transactions.append({
            'sender' : sender,
            'recipient' : recipient,
            'amount' : amount
        })

        return self.last_block['index'] + 1

    @staticmethod
    def hash(block):
        # Hashes a block
        '''
        Creates a SHA-256 hash of a block.

        :param block: <dict> Block
        :return: <str>
        '''
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    @property
    def last_block(self):
        # Return last block in chain
        return self.chain[-1]

    def proof_of_work(self, last_proof):
        '''
        Simple POW algo:
        - Find a number p' such that hash(pp') contains leading 4 zeros, where p is previous p'
        - p is previous proof, p' new proof

        :param last_proof: <int>
        :return: <int>
        '''

        proof = 0
        while self.valid_proof(last_proof, proof) is False:
            proof += 1

        return proof

    @staticmethod
    def valid_proof(last_proof, proof):
        '''
        Validates proof: does hash(last_proof, proof) contain 4 leading zeros?
        :param last_proof: <int> Previous proof
        :param proof: <int> current proof
        :return: <bool> True if correct, false if not
        '''

        guess = '{last_proof}{proof}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == '0000'

    def valid_chain(self, chain):
        '''
        Determines if a given blockchain is valid
        :param chain: <list> a blockchain
        :return: <bool> True if valid, False if not
        '''

        last_block = chain[0]
        current_index = 1

        while current_index < len(chain):
            block = chain[current_index]
            print('{last_block}')
            print('{block}')
            print("\n......\n")

            if block['previous_hash'] != self.hash(last_block):
                return False

            if not self.valid_proof(last_block['proof'], block['proof']):
                return False

            last_block = block
            current_index += 1

        return True

    def resolve_conflicts(self):
        '''
        This is our consensus algo, it resolves conflicts
        by replacing our chain with longest one in the network.

        :return: <bool> True if our chain was replaced, Fallse if not.
        '''

        neighbors = self.nodes
        new_chain = None

        # Only looking for chains longer than ours
        max_length = len(self.chain)

        # Grab and verify the chains from all the nodes in our networks
        for node in neighbors:
            response = requests.get('http://{node}/chain')

            if response.status_code == 200:
                length = response.json()['length']
                chain = response.json()['chain']

                if length > max_length and self.valid_chain(chain):
                    max_length = length
                    new_chain = chain

        if new_chain:
            self.chain = new_chain
            return True

        return False

app = Flask(__name__)

node_identifier = str(uuid4()).replace('-','')

blockchain = Blockchain()

@app.route('/nodes/register', methods=['POST'])
def register_nodes():
    values = request.get_json()

    nodes = values.get('nodes')
    if nodes is None:
        return "Error: Supply a valid list of nodes", 400

    for node in nodes:
        blockchain.register_node(node)

    response = {
        'message': 'New nodes have been added',
        'total_nodes': list(blockchain.nodes)
    }

    return jsonify(response), 201

@app.route('/nodes/resolve', methods=['GET'])
def consensus():
    replaced = blockchain.resolve_conflicts()

    if replaced:
        response = {
            'message', 'Our chain was replaced',
            'new_chain', blockchain.chain
        }
    else:
        response = {
            'message':'Our chain is authoritative',
            'chain': blockchain.chain
        }

        return jsonify(response), 200


@app.route('/mine', methods=['GET'])
def mine():
    last_block = blockchain.last_block
    last_proof = last_block['proof']
    proof = blockchain.proof_of_work(last_proof)

    blockchain.new_transaction(
        sender="0",
        recipient=node_identifier,
        amount = 1,
    )

    block = blockchain.new_block(proof)

    response = {
        'message':'New block forged',
        'index': block['index'],
        'transactions': block['transactions'],
        'proof': block['proof'],
        'previous_hash': block['previous_hash'],
    }

    return jsonify(response), 200

@app.route('/transactions/new', methods=['POST'])
def new_transaction():
    values = request.get_json()

    required = ['sender', 'recipient', 'amount']
    if not all(k in values for k in required):
        return 'Missing values', 400

    index = blockchain.new_transaction(values['sender'], values['recipient'], values['amount'])

    response = {'message', 'Transaction will be added to Block{index}'}
    return jsonify(response), 201

@app.route('/chain', methods=['GET'])
def full_chain():
    response = {
        'chain': blockchain.chain,
        'length': len(blockchain.chain)
    }
    return jsonify(response), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 5001)
