
from modules import *
import os
import numpy as np



class Model():

    def __init__(self,is_training,args,reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self._batch_size = batch_size = args.batch_size
        self.num_skills = num_skills = args.num_skills

        self.num_steps = num_steps = args.num_steps
        #print(num_steps.type)
        input_size = num_skills*2

        inputs = self._input_data = tf.placeholder(tf.int32, shape=(batch_size, num_steps-1))

        self.problems= tf.placeholder(tf.int32, shape=(batch_size, num_steps-1))
        self.target_id = target_id = tf.placeholder(tf.int32, shape=(None))
        self.target_correctness = target_correctness = tf.placeholder(tf.float32, shape=(None))
        self.Q= tf.placeholder(tf.float32, shape=(None,None))
        self.K= tf.placeholder(tf.float32, shape=(None,None))
        self.logits= tf.placeholder(tf.float32, shape=(None,None))
        #input_data = tf.reshape(self._input_data, [-1])
        x=self._input_data
        key_masks = tf.expand_dims(tf.to_float(tf.not_equal(x, 0)), -1)
        with tf.variable_scope("encoder"):
                ## Embedding
               # key_masks = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(x), axis=-1)), -1)
                self.enc, self.lookup = embedding(x,
                                      vocab_size=input_size,
                                      num_units=args.hidden_units,
                                      zero_pad=False,
                                      scale=False,
                                      l2_reg=args.l2_emb,
                                      scope="enc_embed",
                                      with_t=True,
                                      reuse=reuse)
               
                #tf.tile(tf.expand_dims(tf.range(tf.shape(x)[1]), 0), [tf.shape(x)[0], 1])
                ## Positional Encoding
                # if args.pos:
               
                self.enc += embedding(
                    tf.tile(tf.expand_dims(tf.range(tf.shape(x)[1]), 0), [tf.shape(x)[0], 1]),
                                     vocab_size=num_steps-1,
                                     num_units=args.hidden_units,
                                     zero_pad=False,
                                     scale=False,
                                     scope="enc_pe",
                                     l2_reg=args.l2_emb,
                                     reuse=reuse
                        
              )

                self.seq  =  embedding(self.problems,
                                      vocab_size=num_skills,
                                      num_units=args.hidden_units,
                                      zero_pad=True,
                                      scale=False,
                                      l2_reg=args.l2_emb,
                                      scope="que_embed",
                                      reuse=reuse)

                # Dropout
                self.enc *= key_masks
                self.seq *=key_masks
                self.enc = tf.layers.dropout(self.enc,
                                            rate=args.dropout_rate,
                                            training=tf.convert_to_tensor(is_training))


                ## Blocks
                for i in range(args.num_blocks):

                    with tf.variable_scope("num_blocks_{}".format(i)):

                        self.enc, self.QK =  multihead_attention(queries=normalize(self.seq),
                                                        keys=self.enc,
                                                        num_units=args.hidden_units,
                                                        num_heads=args.num_heads,
                                                        dropout_rate=args.dropout_rate,
                                                        is_training=is_training,
                                                        sizeof_V=args.hidden_units,
                                                        with_qk=True,
                                                        causality=True)


                        ### Feed Forward
                        #weights = tf.get_default_graph().get_tensor_by_name(os.path.split(V.name)[0] + '/kernel:0')
                #         #print(weights.shape)
                #         #self.enc = feedforward(self.enc, num_units=[4*args.hidden_units, num_skills])
                        self.enc = feedforward(normalize(self.enc), num_units=[args.hidden_units, args.hidden_units],
                                           dropout_rate=args.dropout_rate, is_training=self.is_training)
                        self.enc *= key_masks
                        self.seq *=key_masks
                        self.enc = normalize(self.enc)

        # #self.QK = tf.matmul(self.Q, tf.transpose(self.K, [0, 2, 1]))
        #
        self.enc = tf.reshape(self.enc, [tf.shape(x)[0] * (self.num_steps-1), args.hidden_units])
        #

        #

        sigmoid_w = tf.get_variable("sigmoid_w", [args.hidden_units, args.num_skills])
        sigmoid_b = tf.get_variable("sigmoid_b", [ args.num_skills])
        #
        self.logits=tf.matmul(self.enc, sigmoid_w)+sigmoid_b
        self.logits = tf.reshape(self.logits, [-1])

        selected_logits = tf.gather(self.logits, self.target_id)
        self._pred = self._pred_values = pred_values = tf.sigmoid(selected_logits)
        #
        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = selected_logits, labels=target_correctness))
        self._cost = cost = loss
        tf.summary.scalar('loss', self._cost)


        print(reuse)
        if reuse is None:
            #tf.summary.scalar('auc', self.auc)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            gvs = self.optimizer.compute_gradients(cost)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            self.train_op = self.optimizer.apply_gradients(capped_gvs)
            #self.train_op = self.optimizer.minimize(self._cost, global_step=self.global_step)



        self.merged = tf.summary.merge_all()
