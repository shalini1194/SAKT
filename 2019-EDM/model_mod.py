from modules import *





class Model():

    def __init__(self,is_training,args,reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self._batch_size = batch_size = args.batch_size
        self.num_skills = num_skills = args.num_skills

        self.num_steps = num_steps = args.num_steps
        #print(num_steps.type)
        input_size = num_skills*2

        inputs = self._input_data = tf.placeholder(tf.int32, shape=(batch_size, num_steps))
        self.questions=tf.placeholder(tf.int32, shape=(batch_size, num_steps))
        self.problem_ids =  tf.placeholder(tf.int32, shape=(batch_size* num_steps))
        self.target_id = target_id = tf.placeholder(tf.int32, shape=(None))
        self.target_correctness = target_correctness = tf.placeholder(tf.float32, shape=(None))

        #input_data = tf.reshape(self._input_data, [-1])
        x=self._input_data
        #print("here")
        #print(len(x[0]))
        #mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)

        with tf.variable_scope("encoder"):
                ## Embedding
                self.enc, que_emb_table = embedding(self.questions,
                                      vocab_size=num_skills,
                                      num_units=args.hidden_units,
                                      zero_pad=True,
                                      scale=True,
                                      l2_reg=args.l2_emb,
                                      scope="que_embed",
                                      with_t=True,
                                      reuse=reuse)

                key_masks = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(self.enc), axis=-1)), -1)
                #print(que_seq.shape)

                #tf.tile(tf.expand_dims(tf.range(tf.shape(x)[1]), 0), [tf.shape(x)[0], 1])
                ## Positional Encoding
                #print(self.enc.shape)
                self.enc += embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.questions)[1]), 0), [tf.shape(self.questions)[0], 1]),
                                      vocab_size=self.num_skills,
                                      num_units=args.hidden_units,
                                      zero_pad=False,
                                      scale=False,
                                      scope="enc_pe",
                                      l2_reg=args.l2_emb,
                                      reuse=reuse
                                      )

                self.enc *= key_masks
                ## Dropout
                self.enc = tf.layers.dropout(self.enc,
                                            rate=args.dropout_rate,
                                            training=tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(args.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ### Multihead Attention
                        self.enc = multihead_attention(queries=normalize(self.enc),
                                                        keys=self.enc,
                                                        num_units=args.hidden_units,
                                                        num_heads=args.num_heads,
                                                        dropout_rate=args.dropout_rate,
                                                        is_training=is_training,
                                                        causality=True)

                        ### Feed Forward
                        self.enc = feedforward(self.enc, num_units=[4*args.hidden_units, args.hidden_units], is_training=is_training)
                self.enc = normalize(self.enc)

        with tf.variable_scope("knowledge_state"):
                self.ks = embedding(x,
                                      vocab_size=2*num_skills,
                                      num_units=args.hidden_units,
                                      zero_pad=True,
                                      scale=True,
                                      l2_reg=args.l2_emb,
                                      scope="enc_embed",
                                      reuse=reuse)

                #print(que_seq.shape)

                #Embedding.write(str(que_emb_table))
                #tf.tile(tf.expand_dims(tf.range(tf.shape(x)[1]), 0), [tf.shape(x)[0], 1])
                ## Positional Encoding
                #print(self.enc.shape)
                self.ks += embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(x)[1]), 0), [tf.shape(x)[0], 1]),
                                      vocab_size=2*self.num_skills,
                                      num_units=args.hidden_units,
                                      zero_pad=False,
                                      scale=False,
                                      scope="ks_pe",
                                      l2_reg=args.l2_emb,
                                      reuse=reuse
                                      )


                ## Dropout
                self.ks = tf.layers.dropout(self.ks,
                                            rate=args.dropout_rate,
                                            training=tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(args.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ### Multihead Attention
                        self.ks = multihead_attention(queries=normalize(self.ks),
                                                        keys=self.ks,
                                                        num_units=args.hidden_units,
                                                        num_heads=args.num_heads,
                                                        dropout_rate=args.dropout_rate,
                                                        is_training=is_training,
                                                        causality=True)

                        ### Feed Forward
                        self.ks = feedforward(self.ks, num_units=[4*args.hidden_units, args.hidden_units],is_training=is_training)
                self.ks = normalize(self.ks)


        self.enc = tf.reshape(self.enc, [tf.shape(self.questions)[0] * self.num_steps, args.hidden_units])


        test_question_emb = tf.nn.embedding_lookup(que_emb_table, self.problem_ids)


        self.ks = tf.reshape(self.ks, [tf.shape(x)[0] * self.num_steps, args.hidden_units])
        X= self.ks


        # W1 = tf.get_variable("W1",[2*args.hidden_units,  args.hidden_units] )
        # b1 = tf.get_variable("b1",[args.hidden_units])
        #
        #
        # X=tf.tanh(tf.matmul(X,W1) + b1)

        sigmoid_w = tf.get_variable("sigmoid_w", [args.hidden_units, self.num_skills])
        sigmoid_b = tf.get_variable("sigmoid_b", [batch_size, self.num_skills])
        X=tf.matmul(X, sigmoid_w)
        # print(X.shape)
        # expanded_tensor = tf.expand_dims(sigmoid_b, -1)
        # print(expanded_tensor.shape)
        # multiples = tf()
        # tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
        # print(tiled_tensor.shape)
        # repeated_tensor = tf.reshape(tiled_tensor, tf.shape(tensor) * 100)
        # print(repeated_tensor.shape)
        X=tf.reshape(X, [-1, self.num_steps, self.num_skills])

        logits = tf.math.add(X , tf.tile(tf.expand_dims(sigmoid_b, 1), [1, X.shape[1], 1])) # shape (3, 3))
        logits = tf.reshape(logits, [-1])


        selected_logits = tf.gather(logits, self.target_id)
        #make prediction
        self._pred = self._pred_values = pred_values = tf.sigmoid(selected_logits)

        # loss function
        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels = target_correctness, logits=selected_logits, ))

        #self._cost = cost = tf.reduce_mean(loss)
        self._cost = cost = loss
        tf.summary.scalar('loss', self._cost)
        reuse=None

        if reuse is None:
            #tf.summary.scalar('auc', self.auc)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            self.train_op = self.optimizer.minimize(self._cost, global_step=self.global_step)
        else:
            tf.summary.scalar('test_auc', self.auc)

        self.merged = tf.summary.merge_all()

    def predict(self, sess, seq):
        return sess.run(self.test_logits,
                        { self.input_seq: seq,  self.is_training: False})
